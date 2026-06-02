import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleOdometry
import cv2
import numpy as np
import os
import math
import time
import logging
from threading import Thread, Lock
from concurrent.futures import ThreadPoolExecutor

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from dt_apriltags import Detector

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class KalmanFilter2D:
    """针对水平投影位置与速度的卡尔曼滤波器"""
    def __init__(self):
        self.dt = 0.04
        self.X = np.zeros((4, 1)) # [x, y, vx, vy]
        self.F = np.array([[1, 0, self.dt, 0],
                           [0, 1, 0, self.dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])
        self.P = np.eye(4) * 1.0
        self.Q = np.eye(4) * 0.1 
        self.R = np.eye(2) * 0.05 

    def predict(self):
        self.X = np.dot(self.F, self.X)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.X[:2].flatten()

    def update(self, z):
        z = np.array(z).reshape(2, 1)
        y = z - np.dot(self.H, self.X)
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.X = self.X + np.dot(K, y)
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)

class UltimateTrackerV9(Node):
    def __init__(self):
        super().__init__('ultimate_tracker_v9')
        
        logger.info("🚀 正在初始化 ROS 2 节点...")
        
        self.is_running = True
        self.vision_executor = ThreadPoolExecutor(max_workers=2)
        self.lock = Lock()
        
        # --- 1. 视觉配置 ---
        self.detector = Detector(families="tag36h11", nthreads=4, quad_decimate=1.0)
        self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
        self.camera_matrix = np.array([[600.0, 0, 400.0], [0, 600.0, 300.0], [0, 0, 1]], dtype=np.float32)
        self.dist_coeffs = np.zeros(5)
        s = 0.8 / 2.0
        self.tag_3d_pts = np.array([[-s, -s, 0], [s, -s, 0], [s, s, 0], [-s, s, 0]], dtype=np.float32)

        # --- 2. 估计器与状态 ---
        self.kf = KalmanFilter2D()
        self.state = "INITIALIZING" 
        self.target_found = False
        self.last_detection_time = time.time()
        self.rel_pos = np.zeros(3) # [Forward, Left, Down]
        self.fps = 0
        self.frame_count = 0
        self.last_fps_time = time.time()
        
        # 核心：姿态解耦参数
        self.current_pitch = 0.0
        self.current_roll = 0.0
        self.debug_frame = None
        
        # --- 3. PID 控制与故障保险 ---
        self.kp_yaw = 2.6
        self.ki_yaw = 0.08
        self.kd_yaw = 0.45
        self.yaw_integral = 0.0
        self.last_yaw_err = 0.0
        self.yaw_speed_cmd = 0.0
        self.failsafe_timeout = 30.0
        
        # --- 4. 航向锁定机制（关键修复） ---
        self.is_yaw_locked = False  # 航向是否已锁定
        self.locked_yaw = 0.0  # 锁定的航向角（弧度）
        self.current_yaw = 0.0  # 当前实际航向（弧度）
        
        # --- 5. PX4 ROS 2 控制接口 ---
        logger.info("📡 创建 PX4 ROS 2 控制接口...")
        
        # Offboard 控制模式
        self.offboard_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', 10)
        
        # 轨迹设定点（用于速度控制）
        self.trajectory_pub = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', 10)
        
        # 飞行器指令（用于解锁和模式切换）
        self.command_pub = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', 10)
        
        # 订阅无人机里程计/姿态信息
        self.odom_sub = self.create_subscription(
            VehicleOdometry,
            '/fmu/out/vehicle_odometry',
            self.odom_callback,
            10
        )
        
        # --- 6. Gazebo 图像订阅 (无需桥接) ---
        logger.info("📷 订阅 Gazebo 图像话题...")
        try:
            from gz.transport13 import Node as GzTransportNode
            from gz.msgs10.image_pb2 import Image as GzImage
            
            self.gz_node = GzTransportNode()
            self.gz_node.subscribe(GzImage, "/slot_machine_camera/front", self.gz_callback)
            logger.info("✅ Gazebo 图像订阅成功：/slot_machine_camera/front")
        except ImportError as e:
            logger.error(f"❌ 无法导入 Gazebo Transport: {e}")
            raise
        
        # 启动序列 - 延迟 3 秒后执行
        logger.info("⏱ 3 秒后启动...")
        self.start_timer = self.create_timer(3.0, self.start_sequence)
        self.sequence_started = False
        
        # 控制循环定时器
        self.control_timer = self.create_timer(0.04, self.control_loop)
        
        # 可视化线程
        self.viz_thread = Thread(target=self.run_visualization)
        self.viz_thread.start()
        
        logger.info("✅ 初始化完成！等待启动...")
        logger.info("=" * 60)

    def odom_callback(self, msg):
        """接收无人机实际航向信息"""
        # VehicleOdometry 中的 q 是四元数 [w, x, y, z]
        q = msg.q
        # 从四元数提取 yaw 角
        self.current_yaw = math.atan2(2.0 * (q[0] * q[3] + q[1] * q[2]), 
                                      1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3]))

    def start_sequence(self):
        """启动序列：解锁 -> Offboard -> 开始搜索"""
        if self.sequence_started:
            return
        self.sequence_started = True
        
        # 取消启动定时器
        self.destroy_timer(self.start_timer)
        
        logger.info("🚁 启动序列开始...")
        
        # 第 1 步：解锁
        self.send_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0, 0.0)
        logger.info("🔓 已发送解锁指令")
        
        # 第 2 步：Offboard 模式
        time.sleep(1)
        self.send_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
        logger.info("🎮 已切换至 Offboard 模式")
        
        # 第 3 步：进入 SEARCHING 状态
        time.sleep(1)
        self.state = "SEARCHING"
        # 重置航向锁定
        self.is_yaw_locked = False
        self.locked_yaw = 0.0
        logger.info("🔍 开始原地旋转搜寻 AprilTag...")
        logger.info("=" * 60)

    def send_vehicle_command(self, command, param1, param2):
        """发送飞行器指令"""
        cmd = VehicleCommand()
        cmd.command = command
        cmd.param1 = param1
        cmd.param2 = param2
        cmd.param3 = 0.0
        cmd.param4 = 0.0
        cmd.param5 = 0.0
        cmd.param6 = 0.0
        cmd.param7 = 0.0
        cmd.target_system = 1
        cmd.target_component = 1
        cmd.source_system = 1
        cmd.source_component = 1
        cmd.from_external = True
        cmd.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.command_pub.publish(cmd)

    def gz_callback(self, gz_msg):
        """Gazebo 图像回调函数"""
        try:
            h, w = gz_msg.height, gz_msg.width
            raw = np.frombuffer(gz_msg.data, dtype=np.uint8).reshape(h, w, -1)
            cv_frame = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR) if raw.shape[2]==3 else cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
            self.vision_executor.submit(self.process_vision, cv_frame)
        except Exception as e:
            logger.error(f"Gazebo 回调错误：{e}")

    def get_body_level_pos(self, cam_x, cam_y, cam_z):
        """
        核心算法：机体坐标系转换
        将相机视野内的 (x, y, z) 映射到水平机体平面，抵消无人机倾斜影响
        """
        cp, sp = math.cos(self.current_pitch), math.sin(self.current_pitch)
        cr, sr = math.cos(self.current_roll), math.sin(self.current_roll)

        real_forward = cam_z * cp + cam_y * sp
        real_left_right = cam_x * cr - cam_y * sr
        
        return real_forward, real_left_right

    def draw_debug_ui(self, frame):
        """完全保留的可视化内容"""
        h, w = frame.shape[:2]
        cv2.drawMarker(frame, (w//2, h//2), (0, 255, 255), cv2.MARKER_CROSS, 40, 1)
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (260, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        color_map = {"SEARCHING": (0, 255, 255), "TRACKING": (0, 255, 0), "LOST": (0, 0, 255), "HOVER": (200, 200, 200)}
        status_color = color_map.get(self.state, (255, 255, 255))
        
        info = [
            (f"STATE: {self.state}", status_color),
            (f"DIST_X: {self.rel_pos[0]:.2f}m", (255, 255, 255)),
            (f"OFF_Y: {self.rel_pos[1]:.2f}m", (255, 255, 255)),
            (f"YAW_ERR: {math.degrees(self.last_yaw_err):.2f}deg", (255, 255, 255)),
            (f"CMD_SPEED: {self.yaw_speed_cmd:.1f}d/s", (0, 255, 0)),
            (f"LOCKED: {math.degrees(self.locked_yaw):.1f}°" if self.is_yaw_locked else "LOCKED: NO", (255, 0, 255)),
            (f"CUR_YAW: {math.degrees(self.current_yaw):.1f}°", (0, 255, 255)),
            (f"FPS: {self.fps}", (200, 200, 200))
        ]
        for i, (text, color) in enumerate(info):
            cv2.putText(frame, text, (15, 30 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def process_vision(self, cv_frame):
        """视觉处理核心"""
        try:
            gray = cv2.cvtColor(cv_frame, cv2.COLOR_BGR2GRAY)
            gray = self.clahe.apply(gray)
            results = self.detector.detect(gray)
            
            pred_xy = self.kf.predict()

            if results:
                best_tag = max(results, key=lambda x: x.decision_margin)
                success, rvec, tvec = cv2.solvePnP(self.tag_3d_pts, 
                                                   best_tag.corners.astype(np.float32), 
                                                   self.camera_matrix, self.dist_coeffs)
                if success:
                    pos = tvec.flatten()
                    real_f, real_l = self.get_body_level_pos(pos[0], pos[1], pos[2])
                    
                    self.kf.update([real_f, real_l])
                    with self.lock:
                        self.rel_pos[0], self.rel_pos[1] = real_f, real_l
                    
                    self.target_found = True
                    self.last_detection_time = time.time()
                    
                    # 关键：从 SEARCHING 切换到 TRACKING 时锁定航向
                    if self.state == "SEARCHING":
                        # 锁定当前航向（这就是发现目标时的朝向）
                        self.locked_yaw = self.current_yaw
                        self.is_yaw_locked = True
                        logger.info(f"🎯 检测到目标！锁定航向：{math.degrees(self.locked_yaw):.1f}°, 当前朝向：{math.degrees(self.current_yaw):.1f}°")
                    
                    self.state = "TRACKING"
                    
                    cv2.polylines(cv_frame, [best_tag.corners.astype(np.int32)], True, (0, 255, 0), 2)
                    cv2.drawFrameAxes(cv_frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.4)
            else:
                time_since_last = time.time() - self.last_detection_time
                if time_since_last < 0.8:
                    # LOST 状态：使用卡尔曼滤波预测位置
                    self.state = "LOST"
                    with self.lock:
                        self.rel_pos[0], self.rel_pos[1] = pred_xy[0], pred_xy[1]
                elif time_since_last < self.failsafe_timeout:
                    # 超过 0.8 秒未检测到，进入 SEARCHING
                    self.target_found = False
                    self.state = "SEARCHING"
                    # 重置航向锁定
                    self.is_yaw_locked = False
                    self.locked_yaw = 0.0
                    self.yaw_integral = 0.0
                    self.last_yaw_err = 0.0
                    self.yaw_speed_cmd = 12.0
                    logger.info(f"🔄 丢失目标！进入 SEARCHING 状态 (丢失时间：{time_since_last:.1f}s)")
                else:
                    # 超过 30 秒，进入 HOVER
                    self.state = "HOVER"

            self.frame_count += 1
            if time.time() - self.last_fps_time > 1.0:
                self.fps, self.frame_count, self.last_fps_time = self.frame_count, 0, time.time()

            self.draw_debug_ui(cv_frame)
            self.debug_frame = cv_frame
        except Exception as e:
            logger.error(f"视觉处理错误：{e}")

    def control_loop(self):
        """
        关键修复：在 TRACKING 状态使用绝对 yaw 角而不是 yawspeed
        """
        try:
            # 发布 Offboard 模式
            offboard_msg = OffboardControlMode()
            offboard_msg.position = False
            offboard_msg.velocity = True
            offboard_msg.acceleration = False
            offboard_msg.attitude = False
            offboard_msg.body_rate = False
            offboard_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
            self.offboard_pub.publish(offboard_msg)
            
            # 发布轨迹设定点
            traj_msg = TrajectorySetpoint()
            traj_msg.position = [float('nan'), float('nan'), float('nan')]
            traj_msg.velocity = [float('nan'), float('nan'), float('nan')]
            traj_msg.acceleration = [float('nan'), float('nan'), float('nan')]
            
            # 关键逻辑
            if self.state == "SEARCHING":
                # 搜寻模式：固定 12 度/秒旋转
                self.yaw_speed_cmd = 12.0
                yaw_speed_rad = math.radians(self.yaw_speed_cmd)
                traj_msg.velocity = [0.0, 0.0, 0.0]
                traj_msg.yawspeed = yaw_speed_rad  # 使用 yawspeed 旋转
                traj_msg.yaw = float('nan')  # yaw 设为 NaN
                logger.debug(f"🔍 SEARCHING: {self.yaw_speed_cmd}°/s, 当前航向：{math.degrees(self.current_yaw):.1f}°")
            
            elif self.state == "HOVER":
                # 悬停
                self.yaw_speed_cmd = 0.0
                traj_msg.velocity = [0.0, 0.0, 0.0]
                traj_msg.yawspeed = 0.0
                traj_msg.yaw = float('nan')

            elif self.state == "TRACKING":
                # 追踪模式：使用 PID 对准
                with self.lock:
                    yaw_err_deg = math.degrees(math.atan2(self.rel_pos[1], self.rel_pos[0]))
                
                dist_factor = np.clip(self.rel_pos[0] / 5.0, 0.4, 1.5)
                dt = 0.04
                self.yaw_integral = np.clip(self.yaw_integral + yaw_err_deg * dt, -5, 5)
                d_err = (yaw_err_deg - self.last_yaw_err) / dt
                
                self.yaw_speed_cmd = (yaw_err_deg * self.kp_yaw * dist_factor) + \
                                     (self.yaw_integral * self.ki_yaw) + \
                                     (d_err * self.kd_yaw)
                
                self.yaw_speed_cmd = np.clip(self.yaw_speed_cmd, -35.0, 35.0)
                if abs(yaw_err_deg) < 0.6:
                    self.yaw_speed_cmd = 0.0
                
                # 关键修复：使用 yawspeed 进行微调，而不是绝对 yaw
                yaw_speed_rad = math.radians(self.yaw_speed_cmd)
                traj_msg.velocity = [0.0, 0.0, 0.0]
                traj_msg.yawspeed = yaw_speed_rad
                traj_msg.yaw = float('nan')  # yaw 设为 NaN，让 yawspeed 生效
                self.last_yaw_err = yaw_err_deg

            elif self.state == "LOST":
                # 丢失模式：使用卡尔曼滤波预测的位置继续追踪
                with self.lock:
                    yaw_err_deg = math.degrees(math.atan2(self.rel_pos[1], self.rel_pos[0]))
                
                dist_factor = np.clip(self.rel_pos[0] / 5.0, 0.4, 1.5)
                dt = 0.04
                self.yaw_integral = np.clip(self.yaw_integral + yaw_err_deg * dt, -5, 5)
                d_err = (yaw_err_deg - self.last_yaw_err) / dt
                
                self.yaw_speed_cmd = (yaw_err_deg * self.kp_yaw * dist_factor) + \
                                     (self.yaw_integral * self.ki_yaw) + \
                                     (d_err * self.kd_yaw)
                
                self.yaw_speed_cmd = np.clip(self.yaw_speed_cmd, -35.0, 35.0)
                if abs(yaw_err_deg) < 0.6:
                    self.yaw_speed_cmd = 0.0
                
                yaw_speed_rad = math.radians(self.yaw_speed_cmd)
                traj_msg.velocity = [0.0, 0.0, 0.0]
                traj_msg.yawspeed = yaw_speed_rad
                traj_msg.yaw = float('nan')
                self.last_yaw_err = yaw_err_deg
            
            traj_msg.timestamp = offboard_msg.timestamp
            self.trajectory_pub.publish(traj_msg)
            
        except Exception as e:
            logger.error(f"控制循环错误：{e}")

    def run_visualization(self):
        """可视化线程"""
        while self.is_running:
            if self.debug_frame is not None:
                cv2.imshow("V9 Tracker (Body-Level Decoupling)", self.debug_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):
                    self.is_running = False
            time.sleep(0.03)

    def destroy_node(self):
        """节点销毁"""
        logger.info("🛑 正在关闭节点...")
        self.is_running = False
        try:
            self.viz_thread.join(timeout=1.0)
        except:
            pass
        self.vision_executor.shutdown(wait=False)
        cv2.destroyAllWindows()
        super().destroy_node()
        logger.info("✅ 节点已关闭")

def main(args=None):
    try:
        logger.info("=" * 60)
        logger.info("🎯 AprilTag 追踪器 V9 - 纯 ROS 2 架构版")
        logger.info("=" * 60)
        
        rclpy.init(args=args)
        uav = UltimateTrackerV9()
        
        logger.info("⏳ 开始运行，3 秒后自动启动...")
        logger.info("=" * 60)
        
        executor = MultiThreadedExecutor()
        executor.add_node(uav)
        executor.spin()
    except KeyboardInterrupt:
        logger.info("\n👋 用户中断")
    except Exception as e:
        logger.error(f"❌ 错误：{e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            if 'uav' in locals():
                uav.destroy_node()
            rclpy.shutdown()
        except:
            pass

if __name__ == "__main__":
    main()