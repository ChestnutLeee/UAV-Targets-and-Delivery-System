#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleOdometry,
    VehicleLocalPosition
)

import cv2
import numpy as np
import math
import time
from enum import Enum
import threading

try:
    from dt_apriltags import Detector
    from gz.transport13 import Node as GzTransportNode
    from gz.msgs10.image_pb2 import Image as GzImage
except ImportError as e:
    print(f"❌ 依赖缺失: {e}")
    exit(1)

# ==========================================
# 1. 状态机定义
# ==========================================
class FlightState(Enum):
    TAKEOFF = 1
    SEARCHING = 2
    APPROACHING = 3
    TRACKING = 4

# ==========================================
# 2. PID控制器类
# ==========================================
class PIDController:
    def __init__(self, kp, ki, kd, integral_limit=0.4):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        self.integral = 0.0
        self.last_error = 0.0
        self.derivative = 0.0
    
    def compute(self, error, dt, use_integral=True, height_scale=1.0):
        # 积分分离逻辑：只有当误差较小时才启用积分
        if use_integral and abs(error) < 0.3:
            self.integral = np.clip(self.integral + error * dt, -self.integral_limit, self.integral_limit)
        else:
            self.integral = 0.0
        
        # 微分项低通滤波
        raw_derivative = (error - self.last_error) / dt if dt > 0 else 0.0
        self.derivative = 0.6 * self.derivative + 0.4 * raw_derivative
        
        self.last_error = error
        return height_scale * (self.kp * error + self.ki * self.integral + self.kd * self.derivative)

# ==========================================
# 3. 统合版无人机主控节点
# ==========================================
class IntegratedDroneMission(Node):
    def __init__(self):
        super().__init__('integrated_drone_mission')
        self.get_logger().info("🚀 统合版无缝状态机已启动")

        # --- 状态与锁 ---
        self.state = FlightState.TAKEOFF
        self.data_lock = threading.Lock()
        
        # 无人机姿态与位置状态
        self.current_alt = 0.0
        self.current_yaw = 0.0
        self.current_pitch = 0.0
        self.current_roll = 0.0
        
        # 视觉检测共享状态
        self.front_tag_detected = False
        self.front_tag_pos = np.zeros(3)  # [X, Y, Z] 相对位置
        self.front_last_seen = 0.0
        
        self.bottom_tag_detected = False
        self.bottom_tag_err = np.zeros(2) # [err_x, err_y] 像素误差比例
        self.bottom_tag_corners = None  # 标签 corners 用于航向角计算
        self.bottom_last_seen = 0.0
        self.height_est = 2.5  # 高度估计

        # --- 视觉配置 ---
        self.detector = Detector(families="tag36h11", nthreads=2, quad_decimate=1.0)
        self.camera_matrix = np.array([[600.0, 0, 400.0], [0, 600.0, 300.0], [0, 0, 1]], dtype=np.float32)
        self.dist_coeffs = np.zeros(5)
        self.tag_size = 0.8  # 标签大小
        s = self.tag_size / 2.0 
        self.tag_3d_pts = np.array([[-s, -s, 0], [s, -s, 0], [s, s, 0], [-s, s, 0]], dtype=np.float32)

        # --- 控制与目标参数 ---
        self.target_altitude = -6.0  # 起飞高度 (NED, -6m = 向上6m)
        self.tracking_altitude = -2.5 # 追踪时高度，降低高度提高稳定性
        self.MAX_ALT_LIMIT = -40.0 # 飞行上限
        self.MIN_ALT_LIMIT = -0.5  # 飞行下限
        
        # 追踪模式 PID (来自 Tracking_Mode，优化稳定性)
        self.pid_vx = PIDController(kp=0.95, ki=0.08, kd=0.25)
        self.pid_vy = PIDController(kp=0.95, ki=0.08, kd=0.25)
        self.pid_yaw = PIDController(kp=0.6, ki=0.0, kd=0.1)
        
        # 逼近模式 PID (来自 Approaching_Mode)
        self.app_pid_vx = PIDController(kp=1.2, ki=0.02, kd=0.15)
        self.app_pid_yaw = PIDController(kp=1.5, ki=0.01, kd=0.10)

        # 平滑输出缓存
        self.cmd_vx, self.cmd_vy, self.cmd_yawspeed = 0.0, 0.0, 0.0
        
        # 控制参数
        self.search_rotation_speed = math.radians(12.0)  # 12度/秒 旋转
        self.approach_target_distance = 1.0  # 目标距离1米
        self.position_tolerance = 0.2  # 位置容差
        self.tag_timeout_front = 1.5  # 前置摄像头标签超时时间
        self.tag_timeout_bottom = 2.0  # 底部摄像头标签超时时间
        self.deadband_threshold = 0.005  # 死区阈值
        self.smooth_factor = 0.85  # 增加平滑因子，提高稳定性
        
        # 速度限制
        self.max_approach_speed = 2.5
        self.min_approach_speed = -0.5
        self.max_yaw_rate = 0.8
        self.min_tracking_speed = 0.5  # 降低最小追踪速度
        self.max_tracking_speed = 2.5  # 降低最大追踪速度，提高稳定性

        # --- ROS 2 通信配置 ---
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT, 
            durability=DurabilityPolicy.TRANSIENT_LOCAL, 
            history=HistoryPolicy.KEEP_LAST, 
            depth=1
        )
        
        self.offboard_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos)
        self.traj_pub = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos)
        self.cmd_pub = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', qos)
        
        self.odom_sub = self.create_subscription(VehicleOdometry, '/fmu/out/vehicle_odometry', self.odom_callback, qos)
        self.pos_sub = self.create_subscription(VehicleLocalPosition, '/fmu/out/vehicle_local_position_v1', self.pos_callback, qos)

        # --- Gazebo 图像订阅 ---
        self.gz_node = GzTransportNode()
        self.gz_node.subscribe(GzImage, "/slot_machine_camera/front", self.gz_front_callback)
        self.gz_node.subscribe(GzImage, "/slot_machine_camera/down", self.gz_bottom_callback)

        # --- 启动定时器 ---
        self.get_logger().info("⏳ 准备就绪，3秒后解锁起飞...")
        self.create_timer(3.0, self.start_sequence)
        self.control_timer = self.create_timer(0.02, self.control_loop) # 50Hz 控制循环，提高响应速度

    # ==========================================
    # 回调函数与传感器数据更新
    # ==========================================
    def pos_callback(self, msg):
        """位置回调函数"""
        self.current_alt = msg.z # NED坐标，向上为负

    def odom_callback(self, msg):
        """里程计回调函数"""
        q = msg.q
        # 四元数转欧拉角
        self.current_yaw = math.atan2(2*(q[0]*q[3]+q[1]*q[2]), 1-2*(q[2]*q[2]+q[3]*q[3]))
        self.current_pitch = math.asin(np.clip(2*(q[0]*q[2]-q[3]*q[1]), -1.0, 1.0))
        self.current_roll = math.atan2(2*(q[0]*q[1]+q[2]*q[3]), 1-2*(q[1]*q[1]-q[2]*q[2]))

    def gz_front_callback(self, gz_msg):
        """前置摄像头视觉处理 (用于 SEARCHING 和 APPROACHING)"""
        if self.state not in [FlightState.SEARCHING, FlightState.APPROACHING]:
            return  # 非关联状态跳过处理，节省算力

        try:
            gray = self._gz_to_gray(gz_msg)
            results = self.detector.detect(gray)

            with self.data_lock:
                if results:
                    # 选择置信度最高的标签
                    best_tag = max(results, key=lambda x: x.decision_margin)
                    success, rvec, tvec = cv2.solvePnP(
                        self.tag_3d_pts, 
                        best_tag.corners.astype(np.float32), 
                        self.camera_matrix, 
                        self.dist_coeffs
                    )
                    if success:
                        # 机体坐标系解耦 (消除俯仰影响)
                        cp, sp = math.cos(self.current_pitch), math.sin(self.current_pitch)
                        cr, sr = math.cos(self.current_roll), math.sin(self.current_roll)
                        cam_x, cam_y, cam_z = tvec.flatten()

                        real_f = cam_z * cp + cam_y * sp
                        real_l = cam_x * cr - cam_y * sr

                        self.front_tag_pos = np.array([real_f, real_l, cam_y])
                        self.front_tag_detected = True
                        self.front_last_seen = time.time()
                else:
                    self.front_tag_detected = False
        except Exception as e:
            self.get_logger().error(f"前置摄像头处理错误: {e}")

    def gz_bottom_callback(self, gz_msg):
        """下置摄像头视觉处理 (用于 TRACKING 和 SEARCHING)"""
        if self.state not in [FlightState.TRACKING, FlightState.APPROACHING, FlightState.SEARCHING]:
            return

        try:
            gray = self._gz_to_gray(gz_msg)
            results = self.detector.detect(gray)

            with self.data_lock:
                if results:
                    # 选择置信度最高的标签
                    tag = max(results, key=lambda x: x.decision_margin)
                    h, w = gray.shape
                    cx, cy = tag.center
                    # 归一化误差 (以中心为0)
                    err_x = (cx - (w / 2)) / (w / 2)
                    err_y = (cy - (h / 2)) / (h / 2)

                    # 高度估算
                    pixel_width = np.linalg.norm(tag.corners[0] - tag.corners[1])
                    if pixel_width > 5:
                        est = (self.camera_matrix[0, 0] * self.tag_size) / pixel_width
                        self.height_est = 0.9 * self.height_est + 0.1 * est

                    self.bottom_tag_err = np.array([err_x, err_y])
                    self.bottom_tag_corners = tag.corners
                    self.bottom_tag_detected = True
                    self.bottom_last_seen = time.time()
                else:
                    self.bottom_tag_detected = False
        except Exception as e:
            self.get_logger().error(f"底部摄像头处理错误: {e}")

    def _gz_to_gray(self, gz_msg):
        """将Gazebo图像转换为灰度图"""
        h, w = gz_msg.height, gz_msg.width
        raw = np.frombuffer(gz_msg.data, dtype=np.uint8).reshape(h, w, -1)
        return cv2.cvtColor(raw, cv2.COLOR_RGB2GRAY) if raw.shape[2] == 3 else raw

    # ==========================================
    # 核心状态机与控制循环
    # ==========================================
    def start_sequence(self):
        """解锁并设置 Offboard，启动初始起飞模式"""
        try:
            self.send_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
            time.sleep(0.5)
            self.send_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
            self.get_logger().info("🔓 已解锁，进入 TAKEOFF 模式")
            self.timer_started = True
        except Exception as e:
            self.get_logger().error(f"启动序列错误: {e}")

    def control_loop(self):
        """核心无缝状态机逻辑 (运行于 50Hz)"""
        now = time.time()
        dt = 0.02

        try:
            # 1. 发布维持 Offboard 的心跳包
            off = OffboardControlMode()
            off.position = (self.state == FlightState.TAKEOFF)
            off.velocity = not off.position
            off.timestamp = int(self.get_clock().now().nanoseconds / 1000)
            self.offboard_pub.publish(off)

            # 2. 准备基础轨迹点
            traj = TrajectorySetpoint()
            traj.position = [float('nan'), float('nan'), float('nan')]
            traj.velocity = [float('nan'), float('nan'), float('nan')]
            traj.yaw = float('nan')
            traj.yawspeed = float('nan')

            # ----------------------------------------------------
            # 🟢 阶段 1: 起飞模式
            # ----------------------------------------------------
            if self.state == FlightState.TAKEOFF:
                traj.position = [0.0, 0.0, self.target_altitude]
                traj.yaw = 0.0
                
                # 判断条件：高度达到目标高度
                if self.current_alt <= self.target_altitude + self.position_tolerance:
                    self.get_logger().info("✅ 到达目标高度，切换至 SEARCHING 模式")
                    self.state = FlightState.SEARCHING
                    self.front_last_seen = now

            # ----------------------------------------------------
            # 🔍 阶段 2: 搜寻模式
            # ----------------------------------------------------
            elif self.state == FlightState.SEARCHING:
                traj.position[2] = self.target_altitude # 维持高度
                traj.velocity = [0.0, 0.0, 0.0]
                traj.yawspeed = self.search_rotation_speed # 旋转搜索
                
                # 判断条件：前置摄像头或下置摄像头发现目标
                with self.data_lock:
                    if self.bottom_tag_detected:
                        # 下置摄像头发现目标，直接进入追踪模式
                        self.get_logger().info("🔽 下置摄像头发现目标，直接进入 TRACKING 模式")
                        self.state = FlightState.TRACKING
                    elif self.front_tag_detected:
                        # 前置摄像头发现目标，进入逼近模式
                        self.get_logger().info("🎯 前置摄像头锁定目标，切换至 APPROACHING 模式")
                        self.state = FlightState.APPROACHING

            # ----------------------------------------------------
            # 🚀 阶段 3: 逼近模式
            # ----------------------------------------------------
            elif self.state == FlightState.APPROACHING:
                traj.position[2] = self.target_altitude 
                
                with self.data_lock:
                    time_since_front = now - self.front_last_seen
                    time_since_bottom = now - self.bottom_last_seen
                    dist_f = self.front_tag_pos[0]
                    
                    # 状态跃迁逻辑
                    if self.bottom_tag_detected:
                        # 优先：底部已捕捉，完美交接
                        self.get_logger().info("🔽 底部摄像头已捕捉，平滑切换至 TRACKING 模式")
                        self.state = FlightState.TRACKING
                    elif time_since_front > self.tag_timeout_front:
                        if dist_f < 1.0: # 距离很近但丢失了，说明跑到下方盲区了
                            self.get_logger().info("盲区预判：目标已到正下方，盲切至 TRACKING 模式")
                            self.state = FlightState.TRACKING
                        else: # 远距离意外丢失
                            self.get_logger().warning("⚠️ 逼近中意外丢失目标，退回 SEARCHING 模式")
                            self.state = FlightState.SEARCHING
                    else:
                        # 持续逼近控制 (PID控制前向速度和偏航角)
                        body_x, body_y = self.front_tag_pos[0], self.front_tag_pos[1]
                        vx_target = self.app_pid_vx.compute(body_x - self.approach_target_distance, dt)
                        yaw_target = self.app_pid_yaw.compute(math.atan2(body_y, body_x), dt)
                        
                        self.cmd_vx = np.clip(vx_target, self.min_approach_speed, self.max_approach_speed)
                        self.cmd_yawspeed = np.clip(yaw_target, -self.max_yaw_rate, self.max_yaw_rate)
                        
                        # 速度转换到全局坐标系
                        traj.velocity = [
                            self.cmd_vx * math.cos(self.current_yaw), 
                            self.cmd_vx * math.sin(self.current_yaw), 
                            0.0
                        ]
                        traj.yawspeed = self.cmd_yawspeed

            # ----------------------------------------------------
            # 🛬 阶段 4: 追踪模式
            # ----------------------------------------------------
            elif self.state == FlightState.TRACKING:
                # 限制追踪高度
                self.tracking_altitude = np.clip(self.tracking_altitude, self.MAX_ALT_LIMIT, self.MIN_ALT_LIMIT)
                traj.position[2] = self.tracking_altitude # 维持追踪高度
                
                with self.data_lock:
                    time_since_bottom = now - self.bottom_last_seen
                    
                    if self.bottom_tag_detected:
                        # 标签被检测到
                        err_x, err_y = self.bottom_tag_err[0], self.bottom_tag_err[1]
                        
                        # 极小死区设置
                        if abs(err_x) < self.deadband_threshold: err_x = 0.0
                        if abs(err_y) < self.deadband_threshold: err_y = 0.0
                        
                        # 动态高度补偿增益
                        h_scale = np.clip(1.0 + 0.2 * (self.height_est - 2.0), 0.8, 2.0)
                        
                        # PID (注意：相机 X 对应无人机左右VY，相机 Y 对应无人机前后VX)
                        target_vx = -self.pid_vy.compute(err_y, dt, use_integral=True, height_scale=h_scale)
                        target_vy = self.pid_vx.compute(err_x, dt, use_integral=True, height_scale=h_scale)
                        
                        # 动态速度限制
                        max_speed = np.clip(0.8 + 0.3 * self.height_est, self.min_tracking_speed, self.max_tracking_speed)
                        
                        # 速度平滑滤波 (提高稳定性)
                        self.cmd_vx = self.smooth_factor * self.cmd_vx + (1 - self.smooth_factor) * np.clip(target_vx, -max_speed, max_speed)
                        self.cmd_vy = self.smooth_factor * self.cmd_vy + (1 - self.smooth_factor) * np.clip(target_vy, -max_speed, max_speed)
                        
                        # 航向角控制
                        if self.bottom_tag_corners is not None:
                            v_side = self.bottom_tag_corners[1] - self.bottom_tag_corners[0]
                            err_yaw = math.atan2(v_side[1], v_side[0])
                            t_yawspeed = np.clip(err_yaw * 0.6, -0.4, 0.4) if abs(err_yaw) > 0.05 else 0.0
                            self.cmd_yawspeed = 0.8 * self.cmd_yawspeed + 0.2 * t_yawspeed
                        else:
                            self.cmd_yawspeed = 0.0
                        
                        traj.velocity = [self.cmd_vx, self.cmd_vy, 0.0]
                        traj.yawspeed = self.cmd_yawspeed
                    elif time_since_bottom > self.tag_timeout_bottom:
                        # 标签完全丢失，退回搜索模式
                        self.get_logger().warning("❌ 底部完全丢失目标！紧急退回 SEARCHING 模式")
                        self.target_altitude = -6.0 # 重新拉高
                        self.state = FlightState.SEARCHING
                    else:
                        # 标签短暂丢失，缓慢减速
                        self.cmd_vx *= 0.92
                        self.cmd_vy *= 0.92
                        self.cmd_yawspeed *= 0.85
                        traj.velocity = [self.cmd_vx, self.cmd_vy, 0.0]
                        traj.yawspeed = self.cmd_yawspeed

            # 发送控制指令
            traj.timestamp = off.timestamp
            self.traj_pub.publish(traj)
        except Exception as e:
            self.get_logger().error(f"控制循环错误: {e}")

    def send_vehicle_command(self, command, param1=0.0, param2=0.0):
        """发送车辆命令"""
        cmd = VehicleCommand()
        cmd.command, cmd.param1, cmd.param2 = command, float(param1), float(param2)
        cmd.target_system, cmd.target_component = 1, 1
        cmd.source_system, cmd.source_component = 1, 1
        cmd.from_external = True
        cmd.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.cmd_pub.publish(cmd)

def main(args=None):
    """主函数"""
    rclpy.init(args=args)
    mission = IntegratedDroneMission()
    # 使用多线程执行器确保 Gazebo 的多个摄像头数据回调不会互相阻塞
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(mission)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        mission.get_logger().info("⏹️ 收到中断信号，正在关闭...")
    finally:
        mission.destroy_node()
        executor.shutdown()
        rclpy.shutdown()

if __name__ == "__main__":
    main()