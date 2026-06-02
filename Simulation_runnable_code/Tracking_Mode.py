import rclpy
from rclpy.node import Node
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint
import cv2
import numpy as np
import apriltag
import time
import math

class UltraStableAdaptiveTracker(Node):

    def __init__(self):
        super().__init__('ultra_stable_adaptive_tracker')

        # 1. PX4 发布器
        self.offboard_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', 10)
        self.traj_pub = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', 10)

        # 2. Gazebo 图像订阅 (gz-transport)
        try:
            from gz.transport13 import Node as GzTransportNode
            from gz.msgs10.image_pb2 import Image as GzImage
            self.gz_node = GzTransportNode()
            self.gz_node.subscribe(GzImage, "/slot_machine_camera/down", self.gz_callback)
        except ImportError:
            self.get_logger().error("❌ 未找到 gz-transport，请确保环境配置正确")

        # 3. AprilTag 检测器
        self.detector = apriltag.Detector(apriltag.DetectorOptions(families="tag36h11"))

        # ==========================================
        # 核心控制参数 (针对高精度对中优化)
        # ==========================================
        self.target_alt = -2.5  # NED 坐标系: -2.5m 为向上
        self.MAX_ALT_LIMIT = -40.0 # 飞行上限
        self.MIN_ALT_LIMIT = -0.5  # 飞行下限
        
        # PID 参数调整
        self.kp = 0.95          # 提高比例增益，增强纠偏响应速度
        self.ki = 0.08          # 提高积分增益，用于消除静差（对准中心的关键）
        self.kd = 0.25          # 提高微分增益，配合积分项抑制过冲
        
        self.deadzone = 0.005   # 极大缩小死区 (0.5% 屏幕比例)，允许微小位移调整
        self.smooth_factor = 0.70 # 降低输出平滑权重 (0.85 -> 0.7)，减少系统滞后

        # 状态变量
        self.err_sum_x = 0.0
        self.err_sum_y = 0.0
        self.last_err_x = 0.0
        self.last_err_y = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.yawspeed = 0.0
        
        self.derivative_x = 0.0
        self.derivative_y = 0.0

        # 视觉/相机内参
        self.tag_size = 0.8   
        self.fx = 600.0       
        self.height_est = 2.5

        self.last_update_time = time.time()
        self.timer = self.create_timer(0.02, self.control_loop)

        self.get_logger().info("------------------------------------------")
        self.get_logger().info("🚀 高精度追踪器已启动 (精准对中优化版)")
        self.get_logger().info("控制模式：PID + 积分分离 + 极小死区")
        self.get_logger().info("------------------------------------------")

    def gz_callback(self, gz_msg):
        try:
            h, w = gz_msg.height, gz_msg.width
            frame = np.frombuffer(gz_msg.data, dtype=np.uint8).reshape(h, w, -1)
            
            if frame.shape[2] == 3:
                cv_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(cv_frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
                cv_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            results = self.detector.detect(gray)
            dt = time.time() - self.last_update_time
            self.last_update_time = time.time()
            if dt < 0.01: dt = 0.02 # 防止时间戳抖动

            # 绘制中心十字准星
            cv2.line(cv_frame, (w//2 - 10, h//2), (w//2 + 10, h//2), (255, 255, 255), 1)
            cv2.line(cv_frame, (w//2, h//2 - 10), (w//2, h//2 + 10), (255, 255, 255), 1)

            if results:
                tag = results[0]
                cx, cy = tag.center

                # 1. 高度估算
                pixel_width = np.linalg.norm(tag.corners[0] - tag.corners[1])
                if pixel_width > 5:
                    est = (self.fx * self.tag_size) / pixel_width
                    self.height_est = 0.9 * self.height_est + 0.1 * est

                # 2. 水平 PID 逻辑
                err_x = (cx - (w / 2)) / (w / 2)
                err_y = (cy - (h / 2)) / (h / 2)

                # 应用精细死区
                if abs(err_x) < self.deadzone: err_x = 0.0
                if abs(err_y) < self.deadzone: err_y = 0.0

                # 动态高度补偿增益
                h_scale = np.clip(1.0 + 0.2 * (self.height_est - 2.0), 0.8, 2.0)

                # --- 积分分离逻辑 ---
                # 只有当误差进入 30% 范围内时才启用积分，防止长距离追踪时的积分饱和
                if abs(err_x) < 0.3:
                    self.err_sum_x = np.clip(self.err_sum_x + err_x * dt, -0.4, 0.4)
                else:
                    self.err_sum_x = 0.0
                
                if abs(err_y) < 0.3:
                    self.err_sum_y = np.clip(self.err_sum_y + err_y * dt, -0.4, 0.4)
                else:
                    self.err_sum_y = 0.0

                # 微分项低通滤波
                raw_d_x = (err_x - self.last_err_x) / dt
                raw_d_y = (err_y - self.last_err_y) / dt
                self.derivative_x = 0.6 * self.derivative_x + 0.4 * raw_d_x
                self.derivative_y = 0.6 * self.derivative_y + 0.4 * raw_d_y

                # 计算目标速度 (VX 对应图像 Y 轴误差，VY 对应图像 X 轴误差)
                target_vx = -(err_y * self.kp * h_scale + self.err_sum_y * self.ki + self.derivative_y * self.kd)
                target_vy = (err_x * self.kp * h_scale + self.err_sum_x * self.ki + self.derivative_x * self.kd)

                # 限制最大速度
                max_speed = np.clip(0.8 + 0.3 * self.height_est, 1.0, 3.5)
                
                # 速度平滑输出 (提升响应灵敏度)
                self.vx = self.smooth_factor * self.vx + (1 - self.smooth_factor) * np.clip(target_vx, -max_speed, max_speed)
                self.vy = self.smooth_factor * self.vy + (1 - self.smooth_factor) * np.clip(target_vy, -max_speed, max_speed)
                
                self.last_err_x, self.last_err_y = err_x, err_y

                # 3. 航向角控制 (可选)
                v_side = tag.corners[1] - tag.corners[0]
                err_yaw = math.atan2(v_side[1], v_side[0])
                t_yawspeed = np.clip(err_yaw * 0.6, -0.4, 0.4) if abs(err_yaw) > 0.05 else 0.0
                self.yawspeed = 0.8 * self.yawspeed + 0.2 * t_yawspeed

                # 可视化增强
                for i in range(4):
                    cv2.line(cv_frame, tuple(tag.corners[i].astype(int)), tuple(tag.corners[(i+1)%4].astype(int)), (0, 255, 0), 2)
                cv2.circle(cv_frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)
            else:
                # 丢失目标时缓慢减速
                self.vx *= 0.92
                self.vy *= 0.92
                self.yawspeed *= 0.85

            # 显示状态
            color = (0, 255, 0) if abs(self.last_err_x) < 0.05 else (0, 0, 255)
            cv2.putText(cv_frame, f"Precision Mode: {'LOCKED' if abs(self.last_err_x) < 0.02 else 'TRACKING'}", (10, 30), 0, 0.6, color, 2)
            cv2.putText(cv_frame, f"X_Err: {self.last_err_x:.3f} Y_Err: {self.last_err_y:.3f}", (10, 60), 0, 0.5, (255, 255, 255), 1)
            
            cv2.imshow("Tracker Control Center", cv_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('w'): self.target_alt -= 0.1
            elif key == ord('s'): self.target_alt += 0.1
            elif key == 27: rclpy.shutdown()

        except Exception as e:
            self.get_logger().error(f"Error in Callback: {e}")

    def control_loop(self):
        self.target_alt = np.clip(self.target_alt, self.MAX_ALT_LIMIT, self.MIN_ALT_LIMIT)

        off = OffboardControlMode()
        off.position, off.velocity = True, True
        off.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_pub.publish(off)

        traj = TrajectorySetpoint()
        traj.position = [float('nan'), float('nan'), float(self.target_alt)]
        traj.velocity = [float(self.vx), float(self.vy), float('nan')]
        traj.yawspeed = float(self.yawspeed)
        traj.timestamp = off.timestamp
        self.traj_pub.publish(traj)

def main():
    rclpy.init()
    node = UltraStableAdaptiveTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()