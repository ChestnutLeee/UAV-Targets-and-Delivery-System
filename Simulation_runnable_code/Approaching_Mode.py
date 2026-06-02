import rclpy
from rclpy.node import Node
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleOdometry
import cv2
import numpy as np
import math
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

try:
    from dt_apriltags import Detector
    print("✅ 成功加载 dt-apriltags 库")
except ImportError:
    print("❌ 错误: 未检测到 dt-apriltags。请执行: pip3 install dt-apriltags")
    exit(1)

class UltimateTrackerV5(Node):
    def __init__(self):
        super().__init__('ultimate_tracker_v5')

        # 1. 修正后的 QoS 配置
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT, 
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST, 
            depth=1
        )
        
        self.offboard_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos)
        self.traj_pub = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos)
        self.odom_sub = self.create_subscription(VehicleOdometry, '/fmu/out/vehicle_odometry', self.odom_callback, qos)

        # 2. 视觉解算参数
        self.camera_matrix = np.array([[600.0, 0, 400.0], [0, 600.0, 300.0], [0, 0, 1]], dtype=np.float32)
        self.dist_coeffs = np.zeros(5)
        s = 0.8 / 2.0 
        self.tag_3d_pts = np.array([[-s, -s, 0], [s, -s, 0], [s, s, 0], [-s, s, 0]], dtype=np.float32)
        
        # 3. EKF 状态向量 [x, y, z, vx, vy, vz]
        self.state = np.zeros((6, 1))
        self.P = np.eye(6) * 1.0
        self.Q = np.eye(6) * 0.05
        self.R = np.eye(3) * 0.1
        
        # 4. 视觉增强与平滑
        self.detector = Detector(families="tag36h11", nthreads=1, quad_decimate=1.0)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        self.smooth_alpha = 0.2
        self.target_pos_filtered = np.array([2.0, 0.0, 0.0])

        # 5. 控制参数
        self.target_dist = 2.0
        self.target_alt = -2.5
        self.current_yaw, self.current_pitch = 0.0, 0.0
        self.pid_vx = {'kp': 1.2, 'ki': 0.02, 'kd': 0.15, 'ff': 0.5, 'itg': 0.0, 'last_e': 0.0}
        self.pid_yaw = {'kp': 1.5, 'ki': 0.01, 'kd': 0.10, 'ff': 0.0, 'itg': 0.0, 'last_e': 0.0}
        self.vx, self.yawspeed = 0.0, 0.0
        self.last_time = self.get_clock().now()
        self.lost_timer = 0.0

        # 6. Gz 图像订阅
        try:
            from gz.transport13 import Node as GzTransportNode
            from gz.msgs10.image_pb2 import Image as GzImage
            self.gz_node = GzTransportNode()
            self.gz_node.subscribe(GzImage, "/slot_machine_camera/front", self.gz_callback)
            self.get_logger().info("🚀 V5.1 启动成功：EKF 模式已激活")
        except:
            self.get_logger().error("Gz Transport 异常")

        self.create_timer(0.02, self.control_loop)

    def odom_callback(self, msg):
        q = msg.q
        self.current_yaw = math.atan2(2*(q[0]*q[3]+q[1]*q[2]), 1-2*(q[2]*q[2]+q[3]*q[3]))
        self.current_pitch = math.asin(np.clip(2*(q[0]*q[2]-q[3]*q[1]), -1.0, 1.0))

    def ekf_predict(self, dt):
        F = np.eye(6)
        for i in range(3): F[i, i+3] = dt
        self.state = F @ self.state
        self.P = F @ self.P @ F.T + self.Q

    def ekf_update(self, z):
        H = np.zeros((3, 6))
        H[:3, :3] = np.eye(3)
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.state = self.state + K @ (z.reshape(3,1) - H @ self.state)
        self.P = (np.eye(6) - K @ H) @ self.P

    def run_pid(self, p, error, dt, ff):
        p['itg'] = np.clip(p['itg'] + error * dt, -0.5, 0.5)
        deriv = (error - p['last_e']) / dt if dt > 0 else 0.0
        p['last_e'] = error
        return p['kp'] * error + p['ki'] * p['itg'] + p['kd'] * deriv + p['ff'] * ff

    def gz_callback(self, gz_msg):
        h, w = gz_msg.height, gz_msg.width
        raw = np.frombuffer(gz_msg.data, dtype=np.uint8).reshape(h, w, -1)
        cv_frame = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR) if raw.shape[2]==3 else cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
        
        # 1. 预处理
        gray = cv2.cvtColor(cv_frame, cv2.COLOR_BGR2GRAY)
        gray = self.clahe.apply(gray)
        
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds / 1e9
        self.last_time = now
        self.ekf_predict(dt)

        results = self.detector.detect(gray)
        found = False
        if results:
            success, rvec, tvec = cv2.solvePnP(self.tag_3d_pts, results[0].corners.astype(np.float32), 
                                               self.camera_matrix, self.dist_coeffs)
            if success:
                self.ekf_update(tvec.flatten())
                self.lost_timer = 0.0
                found = True

        if not found: self.lost_timer += dt
        
        if self.lost_timer < 1.5:
            # 状态提取与平滑
            raw_target = np.array([self.state[2,0], self.state[0,0], self.state[1,0]])
            self.target_pos_filtered = self.smooth_alpha * raw_target + (1 - self.smooth_alpha) * self.target_pos_filtered
            
            body_x, body_y = self.target_pos_filtered[0] * math.cos(self.current_pitch), self.target_pos_filtered[1]
            self.vx = np.clip(self.run_pid(self.pid_vx, body_x - self.target_dist, dt, self.state[5,0] if found else 0.0), -0.5, 2.2)
            self.yawspeed = self.run_pid(self.pid_yaw, math.atan2(body_y, body_x), dt, 0.0)
            
            cv2.putText(cv_frame, f"{'LOCKED' if found else 'EKF'} Dist: {body_x:.2f}m", (10, 30), 0, 0.7, (0, 255, 0), 2)
        else:
            self.vx, self.yawspeed = self.vx * 0.8, self.yawspeed * 0.5

        cv2.imshow("V5.1 Ultimate Tracker", cv_frame)
        cv2.waitKey(1)

    def control_loop(self):
        off = OffboardControlMode()
        off.position, off.velocity, off.timestamp = False, True, int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_pub.publish(off)

        traj = TrajectorySetpoint()
        traj.position = [float('nan'), float('nan'), float(self.target_alt)]
        traj.velocity = [float(self.vx * math.cos(self.current_yaw)), float(self.vx * math.sin(self.current_yaw)), float('nan')]
        traj.yawspeed = float(self.yawspeed)
        traj.timestamp = off.timestamp
        self.traj_pub.publish(traj)

def main():
    rclpy.init()
    node = UltimateTrackerV5()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()