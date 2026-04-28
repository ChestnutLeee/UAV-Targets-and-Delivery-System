#!/usr/bin/env python3
"""
基于机器视觉的无人机动态靶标精准识别与投送系统 —— 实机控制节点（集成 EVSR 超分）
硬件平台: Jetson Orin Nano + Holybro Pixhawk4 + 双 CIS 摄像头
依赖: ROS2 (Humble), MAVROS, dt_apriltags, OpenCV, numba, pycuda, tensorrt
"""

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode, CommandLong
import cv2
import numpy as np
import math
import time
from enum import Enum
import threading
import os

# -------------------------- GPU 加速库 --------------------------
try:
    from numba import cuda as numba_cuda
    NUMBA_CUDA_AVAILABLE = numba_cuda.is_available()
except ImportError:
    NUMBA_CUDA_AVAILABLE = False

try:
    import pycuda.autoinit
    import pycuda.driver as cuda
    import tensorrt as trt
    PYCUDA_AVAILABLE = True
    TENSORRT_AVAILABLE = True
except ImportError as e:
    PYCUDA_AVAILABLE = False
    TENSORRT_AVAILABLE = False

# -------------------------- 状态机 --------------------------
class FlightState(Enum):
    TAKEOFF = 1
    SEARCHING = 2
    APPROACHING = 3
    TRACKING = 4
    DROP = 5

# -------------------------- PID 控制器 --------------------------
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
        if use_integral and abs(error) < 0.3:
            self.integral = np.clip(self.integral + error * dt, -self.integral_limit, self.integral_limit)
        else:
            self.integral = 0.0
        raw_derivative = (error - self.last_error) / dt if dt > 0 else 0.0
        self.derivative = 0.6 * self.derivative + 0.4 * raw_derivative
        self.last_error = error
        return height_scale * (self.kp * error + self.ki * self.integral + self.kd * self.derivative)

    def reset(self):
        self.integral = 0.0
        self.last_error = 0.0
        self.derivative = 0.0

# -------------------------- 简单卡尔曼滤波器 (3D位置) --------------------------
class SimpleKalman3D:
    def __init__(self, dt=0.05, q=0.01, r=0.5):
        self.dt = dt
        self.F = np.eye(6)
        self.F[0,3] = dt; self.F[1,4] = dt; self.F[2,5] = dt
        self.H = np.zeros((3,6))
        self.H[0,0] = 1; self.H[1,1] = 1; self.H[2,2] = 1
        self.Q = np.eye(6) * q
        self.R = np.eye(3) * r
        self.P = np.eye(6) * 10.0
        self.x = np.zeros((6,1))    # [x,y,z,vx,vy,vz]
        self.initialized = False

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        if not self.initialized:
            self.x[0:3] = z.reshape((3,1))
            self.initialized = True
            return self.x[0:3].flatten()
        self.predict()
        y = z.reshape((3,1)) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P
        return self.x[0:3].flatten()

# -------------------------- CUDA 核函数（可选） --------------------------
if NUMBA_CUDA_AVAILABLE:
    @numba_cuda.jit
    def cuda_image_processing(img, output):
        i, j = numba_cuda.grid(2)
        if i < img.shape[0] and j < img.shape[1]:
            pixel = img[i, j]
            gray = 0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2]
            gray = min(255, max(0, 1.2 * (gray - 128) + 128))
            output[i, j] = gray

# -------------------------- EVSR 超分引擎 --------------------------
class EDSR_TensorRT:
    """轻量版 EDSR TensorRT 推理器（与 EVSR 相同）"""
    def __init__(self, engine_path, input_size=(480, 640)):
        self.input_size = input_size  # H, W
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # 获取绑定信息
        self.input_idx = self.engine.get_binding_index('input')
        self.output_idx = self.engine.get_binding_index('output')
        self.input_shape = (1, 3, *self.input_size)
        self.output_shape = (1, 3, self.input_size[0]*2, self.input_size[1]*2)

        # 分配显存
        self.d_input = cuda.mem_alloc(int(np.prod(self.input_shape) * np.float32().itemsize))
        self.d_output = cuda.mem_alloc(int(np.prod(self.output_shape) * np.float32().itemsize))
        self.bindings = [int(self.d_input), int(self.d_output)]
        self.stream = cuda.Stream()

        # pinned memory 加速传输
        self.h_input = cuda.pagelocked_empty(self.input_shape, dtype=np.float32)
        self.h_output = cuda.pagelocked_empty(self.output_shape, dtype=np.float32)

    def upscale(self, img_bgr):
        """输入 BGR 图像 (uint8)，返回 2倍超分 BGR 图像"""
        h, w = self.input_size
        img = cv2.resize(img_bgr, (w, h))
        # 预处理：BGR→RGB，HWC→CHW，归一化
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_chw = np.transpose(img_rgb, (2, 0, 1)).astype(np.float32) / 255.0
        np.copyto(self.h_input[0], img_chw)

        # 异步推理
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        self.stream.synchronize()

        # 后处理
        out_chw = self.h_output[0] * 255.0
        out_chw = np.clip(out_chw, 0, 255).astype(np.uint8)
        out_rgb = np.transpose(out_chw, (1, 2, 0))
        out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
        return out_bgr

# -------------------------- 主控制节点 --------------------------
class IntegratedDroneMission(Node):
    def __init__(self):
        super().__init__('integrated_drone_mission')
        self.get_logger().info("🛸 实机系统启动 - 集成 EVSR 超分")

        # ---------- 状态与锁 ----------
        self.state = FlightState.TAKEOFF
        self.data_lock = threading.Lock()
        self.camera_running = True

        self.current_alt = 0.0
        self.current_yaw = 0.0
        self.current_pitch = 0.0
        self.current_roll = 0.0
        self.current_pos = [0.0, 0.0, 0.0]
        self.vehicle_state = State()

        # 视觉检测共享变量
        self.front_tag_detected = False
        self.front_tag_pos = np.zeros(3)
        self.front_last_seen = 0.0

        self.bottom_tag_detected = False
        self.bottom_tag_err = np.zeros(2)
        self.bottom_tag_corners = None
        self.bottom_last_seen = 0.0
        self.height_est = 2.5
        self.raw_drone_pos = np.zeros(3)
        self.drone_pos_filtered = np.zeros(3)

        # ---------- EVSR 超分初始化 ----------
        self.evsr_enabled = False
        engine_path = '/home/jetson/edsr_x2_fp16.engine'  # 请修改为实际路径
        if os.path.exists(engine_path) and PYCUDA_AVAILABLE and TENSORRT_AVAILABLE:
            try:
                self.evsr = EDSR_TensorRT(engine_path, input_size=(480, 640))
                self.evsr_enabled = True
                self.get_logger().info("✅ EVSR 超分引擎已启用")
            except Exception as e:
                self.get_logger().warning(f"EVSR 加载失败: {e}，回退到原始分辨率")
        else:
            self.get_logger().info("未找到 EVSR 引擎，运行在原始分辨率")

        # ---------- 视觉配置 ----------
        from dt_apriltags import Detector
        self.detector = Detector(families="tag36h11", nthreads=4, quad_decimate=0.8,
                                 quad_sigma=0.0, refine_edges=1, decode_sharpening=0.25, debug=0)

        # 相机内参 (原始 640x480)
        fx = fy = 402.0
        cx, cy = 320.0, 240.0
        self.camera_matrix_orig = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        self.dist_coeffs = np.zeros(5)

        # 根据超分状态选择内参
        if self.evsr_enabled:
            self.camera_matrix = self.camera_matrix_orig * 2.0
            self.camera_matrix[2,2] = 1.0
            self.img_width, self.img_height = 1280, 960
        else:
            self.camera_matrix = self.camera_matrix_orig.copy()
            self.img_width, self.img_height = 640, 480

        # 标签尺寸
        self.tag_size = 0.2
        s = self.tag_size / 2.0
        self.tag_3d_pts = np.array([[-s, -s, 0], [s, -s, 0], [s, s, 0], [-s, s, 0]], dtype=np.float32)

        # 相机外参
        self.angle_front = math.radians(45)
        c, s_sq = math.cos(-self.angle_front), math.sin(-self.angle_front)
        self.R_front_cam2body = np.array([[c, 0, s_sq], [0, 1, 0], [-s_sq, 0, c]])
        self.R_bottom_cam2body = np.eye(3)

        # ---------- 控制参数 ----------
        self.target_altitude = 2.5
        self.search_rotation_speed = math.radians(15.0)
        self.approach_target_distance = 1.0
        self.tracking_altitude = 1.5
        self.MAX_ALT_LIMIT = 5.0
        self.MIN_ALT_LIMIT = 0.8

        self.pid_vx = PIDController(kp=0.95, ki=0.08, kd=0.25)
        self.pid_vy = PIDController(kp=0.95, ki=0.08, kd=0.25)
        self.pid_yaw = PIDController(kp=0.6, ki=0.0, kd=0.1)
        self.app_pid_vx = PIDController(kp=1.2, ki=0.02, kd=0.15)
        self.app_pid_yaw = PIDController(kp=1.5, ki=0.01, kd=0.10)
        self.pid_alt = PIDController(kp=0.8, ki=0.1, kd=0.2, integral_limit=0.3)

        self.cmd_vx = 0.0; self.cmd_vy = 0.0; self.cmd_yawspeed = 0.0; self.cmd_vz = 0.0

        self.position_tolerance = 0.2
        self.tag_timeout_front = 1.0
        self.tag_timeout_bottom = 1.5
        self.deadband_threshold = 0.005
        self.smooth_factor = 0.8
        self.max_approach_speed = 3.0
        self.min_approach_speed = -0.5
        self.max_yaw_rate = 1.0
        self.min_tracking_speed = 0.5
        self.max_tracking_speed = 3.0
        self.max_vert_speed = 1.0

        self.drop_ready_time = 0.0
        self.drop_required_duration = 3.0
        self.drop_position_tolerance = 0.1
        self.drop_yaw_tolerance = math.radians(5)

        self.vision_lost_time = 0.0
        self.max_vision_lost_duration = 15.0

        # 卡尔曼
        self.kf = SimpleKalman3D(dt=0.05, q=0.01, r=0.5)

        # ---------- ROS 2 通信 ----------
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         history=HistoryPolicy.KEEP_LAST, depth=1)

        self.local_pos_pub = self.create_publisher(PoseStamped, '/mavros/setpoint_position/local', 10)
        self.cmd_vel_pub = self.create_publisher(TwistStamped, '/mavros/setpoint_velocity/cmd_vel', 10)
        self.vision_pose_pub = self.create_publisher(PoseStamped, '/mavros/vision_pose/pose', 10)

        self.state_sub = self.create_subscription(State, '/mavros/state', self.state_callback, 10)
        self.local_pos_sub = self.create_subscription(PoseStamped, '/mavros/local_position/pose', self.pos_callback, 10)

        # ---------- 摄像头初始化 ----------
        gst_pipeline_front = (
            'nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=640, height=480, framerate=30/1 ! '
            'nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
        )
        gst_pipeline_bottom = (
            'nvarguscamerasrc sensor-id=1 ! video/x-raw(memory:NVMM), width=640, height=480, framerate=30/1 ! '
            'nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
        )

        self.front_camera = cv2.VideoCapture(gst_pipeline_front, cv2.CAP_GSTREAMER)
        self.bottom_camera = cv2.VideoCapture(gst_pipeline_bottom, cv2.CAP_GSTREAMER)

        if not self.front_camera.isOpened():
            self.get_logger().warning("前置摄像头 GStreamer 失败，尝试 USB 0")
            self.front_camera = cv2.VideoCapture(0)
            self.front_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.front_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not self.bottom_camera.isOpened():
            self.get_logger().warning("底部摄像头 GStreamer 失败，尝试 USB 1")
            self.bottom_camera = cv2.VideoCapture(1)
            self.bottom_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.bottom_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # ---------- 起飞准备 ----------
        self.timer_started = False
        self.offboard_active = False
        self.create_timer(3.0, self.start_sequence)
        self.control_timer = self.create_timer(0.01, self.control_loop)

        self.front_camera_thread = threading.Thread(target=self.front_camera_loop, daemon=True)
        self.bottom_camera_thread = threading.Thread(target=self.bottom_camera_loop, daemon=True)
        self.front_camera_thread.start()
        self.bottom_camera_thread.start()

        self.emergency_land_triggered = False
        self.last_offboard_heartbeat = time.time()
        self.watchdog_timer = self.create_timer(0.5, self.watchdog_check)
        self.last_vision_pub_time = 0

        self.get_logger().info("✅ 安全系统初始化完成，等待解锁指令")

    # ---------- 回调 ----------
    def state_callback(self, msg):
        self.vehicle_state = msg

    def pos_callback(self, msg):
        self.current_pos = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        self.current_alt = msg.pose.position.z
        q = msg.pose.orientation
        self.current_yaw = math.atan2(2*(q.x*q.z+q.y*q.w), 1-2*(q.z*q.z+q.w*q.w))
        self.current_pitch = math.asin(np.clip(2*(q.x*q.z-q.w*q.y), -1.0, 1.0))
        self.current_roll = math.atan2(2*(q.x*q.y+q.z*q.w), 1-2*(q.y*q.y+q.z*q.z))

    # ---------- 图像处理（通用，不再强制 resize） ----------
    def process_image(self, frame):
        h, w = frame.shape[:2]
        if NUMBA_CUDA_AVAILABLE:
            d_frame = numba_cuda.to_device(frame)
            d_output = numba_cuda.device_array((h, w), dtype=np.uint8)
            threadsperblock = (16, 16)
            blockspergrid_x = math.ceil(h / threadsperblock[0])
            blockspergrid_y = math.ceil(w / threadsperblock[1])
            cuda_image_processing[(blockspergrid_x, blockspergrid_y), threadsperblock](d_frame, d_output)
            gray = d_output.copy_to_host()
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.convertScaleAbs(gray, alpha=1.2, beta=-30)
        return gray

    # ---------- 前视摄像头循环 ----------
    def front_camera_loop(self):
        while rclpy.ok() and self.camera_running:
            if self.state not in [FlightState.SEARCHING, FlightState.APPROACHING]:
                time.sleep(0.1)
                continue
            ret, frame = self.front_camera.read()
            if not ret:
                time.sleep(0.01)
                continue
            try:
                # EVSR 超分增强
                if self.evsr_enabled:
                    frame = self.evsr.upscale(frame)
                gray = self.process_image(frame)
                results = self.detector.detect(gray)
                with self.data_lock:
                    if results:
                        best_tag = max(results, key=lambda x: x.decision_margin)
                        success, rvec, tvec = cv2.solvePnP(self.tag_3d_pts,
                                                           best_tag.corners.astype(np.float32),
                                                           self.camera_matrix, self.dist_coeffs)
                        if success:
                            t_cam = tvec.flatten()
                            t_body = self.R_front_cam2body @ t_cam
                            self.front_tag_pos = t_body
                            self.front_tag_detected = True
                            self.front_last_seen = time.time()
                    else:
                        self.front_tag_detected = False
            except Exception as e:
                self.get_logger().error(f"前置摄像头错误: {e}")
            time.sleep(0.01)

    # ---------- 底部摄像头循环 ----------
    def bottom_camera_loop(self):
        while rclpy.ok() and self.camera_running:
            if self.state not in [FlightState.SEARCHING, FlightState.APPROACHING, FlightState.TRACKING]:
                time.sleep(0.1)
                continue
            ret, frame = self.bottom_camera.read()
            if not ret:
                time.sleep(0.01)
                continue
            try:
                # EVSR 超分增强
                if self.evsr_enabled:
                    frame = self.evsr.upscale(frame)
                gray = self.process_image(frame)
                results = self.detector.detect(gray)
                h, w = gray.shape
                with self.data_lock:
                    if results:
                        tag = max(results, key=lambda x: x.decision_margin)
                        cx, cy = tag.center
                        err_x = (cx - w/2) / (w/2)
                        err_y = (cy - h/2) / (h/2)

                        pixel_width = np.linalg.norm(tag.corners[0] - tag.corners[1])
                        if pixel_width > 5:
                            est = (self.camera_matrix[0,0] * self.tag_size) / pixel_width
                            self.height_est = 0.9 * self.height_est + 0.1 * est

                        success, rvec, tvec = cv2.solvePnP(self.tag_3d_pts,
                                                           tag.corners.astype(np.float32),
                                                           self.camera_matrix, self.dist_coeffs)
                        if success:
                            t_cam = tvec.flatten()
                            t_body = self.R_bottom_cam2body @ t_cam
                            cp, sp = math.cos(self.current_pitch), math.sin(self.current_pitch)
                            cr, sr = math.cos(self.current_roll), math.sin(self.current_roll)
                            cy, sy = math.cos(self.current_yaw), math.sin(self.current_yaw)
                            R_body2enu = np.array([
                                [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                                [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                                [-sp,   cp*sr,            cp*cr]
                            ])
                            drone_pos = -R_body2enu @ t_body
                            filtered = self.kf.update(drone_pos)
                            self.raw_drone_pos = drone_pos
                            self.drone_pos_filtered = filtered

                            now = time.time()
                            if now - self.last_vision_pub_time >= 0.05:
                                vision_pose = PoseStamped()
                                vision_pose.header.stamp = self.get_clock().now().to_msg()
                                vision_pose.header.frame_id = "map"
                                vision_pose.pose.position.x = filtered[0]
                                vision_pose.pose.position.y = filtered[1]
                                vision_pose.pose.position.z = filtered[2]
                                vision_pose.pose.orientation.w = 1.0
                                self.vision_pose_pub.publish(vision_pose)
                                self.last_vision_pub_time = now
                                self.vision_lost_time = 0.0

                        self.bottom_tag_err = np.array([err_x, err_y])
                        self.bottom_tag_corners = tag.corners
                        self.bottom_tag_detected = True
                        self.bottom_last_seen = time.time()
                    else:
                        self.bottom_tag_detected = False
            except Exception as e:
                self.get_logger().error(f"底部摄像头错误: {e}")
            time.sleep(0.01)

    # ---------- 启动序列 ----------
    def start_sequence(self):
        self.get_logger().info("正在请求 Offboard 模式并解锁...")
        if not self.request_offboard_mode():
            self.get_logger().error("无法进入 Offboard")
            return
        time.sleep(1.0)
        if not self.arm_vehicle():
            self.get_logger().error("解锁失败")
            return
        self.offboard_active = True
        self.timer_started = True
        self.get_logger().info("✅ 解锁完成，进入 TAKEOFF")

    def request_offboard_mode(self):
        cli = self.create_client(SetMode, '/mavros/set_mode')
        for _ in range(5):
            if cli.wait_for_service(timeout_sec=1.0):
                break
            self.get_logger().info('等待 set_mode 服务...')
        else:
            return False
        req = SetMode.Request()
        req.custom_mode = 'OFFBOARD'
        cli.call_async(req)
        time.sleep(0.5)
        return True

    def arm_vehicle(self):
        cli = self.create_client(CommandBool, '/mavros/cmd/arming')
        for _ in range(5):
            if cli.wait_for_service(timeout_sec=1.0):
                break
            self.get_logger().info('等待 arming 服务...')
        else:
            return False
        req = CommandBool.Request()
        req.value = True
        cli.call_async(req)
        time.sleep(0.5)
        return True

    def request_mode(self, custom_mode):
        cli = self.create_client(SetMode, '/mavros/set_mode')
        if not cli.wait_for_service(timeout_sec=1.0):
            return
        req = SetMode.Request()
        req.custom_mode = custom_mode
        cli.call_async(req)

    def trigger_drop(self):
        self.get_logger().info("🎯 投放指令已发送！")
        cli = self.create_client(CommandLong, '/mavros/cmd/command')
        if not cli.wait_for_service(timeout_sec=1.0):
            return
        req = CommandLong.Request()
        req.command = 183
        req.param1 = 5
        req.param2 = 1900
        req.confirmation = 0
        cli.call_async(req)
        time.sleep(0.5)
        req.param2 = 1100
        cli.call_async(req)

    def watchdog_check(self):
        if time.time() - self.last_offboard_heartbeat > 0.6:
            self.get_logger().error("⏰ 看门狗超时，紧急降落！")
            self.trigger_emergency_land()

    def trigger_emergency_land(self):
        if self.emergency_land_triggered:
            return
        self.emergency_land_triggered = True
        self.control_timer.cancel()
        self.request_mode("AUTO.LAND")
        self.get_logger().error("🚨 触发紧急降落！")

    # ---------- 核心控制循环（保持不变） ----------
    def control_loop(self):
        if self.emergency_land_triggered or not self.timer_started:
            return
        now = time.time()
        dt = 0.01

        if self.vehicle_state.mode != "OFFBOARD" and not self.emergency_land_triggered:
            self.get_logger().warn(f"退出 Offboard 模式，尝试恢复")
            if not self.request_offboard_mode():
                self.trigger_emergency_land()
                return

        if self.state not in [FlightState.TAKEOFF, FlightState.DROP]:
            self.vision_lost_time += dt if not (self.bottom_tag_detected or self.front_tag_detected) else -0.1
            self.vision_lost_time = max(0, self.vision_lost_time)
            if self.vision_lost_time > self.max_vision_lost_duration:
                self.get_logger().error("视觉丢失超时，返航")
                self.trigger_emergency_land()
                return

        current_target_alt = self.target_altitude
        if self.state in [FlightState.TRACKING, FlightState.APPROACHING]:
            current_target_alt = self.tracking_altitude
        alt_error = current_target_alt - self.current_alt
        self.cmd_vz = np.clip(self.pid_alt.compute(alt_error, dt), -self.max_vert_speed, self.max_vert_speed)

        try:
            if self.state == FlightState.TAKEOFF:
                twist = TwistStamped()
                twist.header.stamp = self.get_clock().now().to_msg()
                twist.header.frame_id = "base_link"
                twist.twist.linear.z = self.cmd_vz
                self.cmd_vel_pub.publish(twist)
                self.last_offboard_heartbeat = time.time()
                if abs(self.current_alt - self.target_altitude) < self.position_tolerance:
                    self.get_logger().info("到达目标高度 → SEARCHING")
                    self._reset_all_pid()
                    self.state = FlightState.SEARCHING

            elif self.state == FlightState.SEARCHING:
                twist = TwistStamped()
                twist.header.stamp = self.get_clock().now().to_msg()
                twist.header.frame_id = "base_link"
                twist.twist.angular.z = self.search_rotation_speed
                twist.twist.linear.z = self.cmd_vz
                self.cmd_vel_pub.publish(twist)
                self.last_offboard_heartbeat = time.time()
                with self.data_lock:
                    if self.bottom_tag_detected:
                        self.get_logger().info("底部发现目标 → TRACKING")
                        self._reset_all_pid()
                        self.state = FlightState.TRACKING
                    elif self.front_tag_detected:
                        self.get_logger().info("前方发现目标 → APPROACHING")
                        self._reset_all_pid()
                        self.state = FlightState.APPROACHING

            elif self.state == FlightState.APPROACHING:
                with self.data_lock:
                    time_since_front = now - self.front_last_seen
                    dist_f = self.front_tag_pos[0]
                    if self.bottom_tag_detected:
                        self.get_logger().info("底部已捕捉 → TRACKING")
                        self._reset_all_pid()
                        self.state = FlightState.TRACKING
                    elif time_since_front > self.tag_timeout_front:
                        if dist_f < 1.0:
                            self.get_logger().info("盲区预判 → TRACKING")
                            self._reset_all_pid()
                            self.state = FlightState.TRACKING
                        else:
                            self.get_logger().warning("丢失目标 → SEARCHING")
                            self._reset_all_pid()
                            self.state = FlightState.SEARCHING
                    else:
                        body_x, body_y = self.front_tag_pos[0], self.front_tag_pos[1]
                        vx_target = self.app_pid_vx.compute(body_x - self.approach_target_distance, dt)
                        yaw_target = self.app_pid_yaw.compute(math.atan2(body_y, body_x), dt)
                        self.cmd_vx = np.clip(vx_target, self.min_approach_speed, self.max_approach_speed)
                        self.cmd_yawspeed = np.clip(yaw_target, -self.max_yaw_rate, self.max_yaw_rate)
                        twist = TwistStamped()
                        twist.header.stamp = self.get_clock().now().to_msg()
                        twist.header.frame_id = "base_link"
                        twist.twist.linear.x = self.cmd_vx
                        twist.twist.angular.z = self.cmd_yawspeed
                        twist.twist.linear.z = self.cmd_vz
                        self.cmd_vel_pub.publish(twist)
                        self.last_offboard_heartbeat = time.time()

            elif self.state in [FlightState.TRACKING, FlightState.DROP]:
                with self.data_lock:
                    time_since_bottom = now - self.bottom_last_seen
                    if self.bottom_tag_detected:
                        err_x, err_y = self.bottom_tag_err
                        if abs(err_x) < self.deadband_threshold: err_x = 0.0
                        if abs(err_y) < self.deadband_threshold: err_y = 0.0
                        h_scale = np.clip(1.0 + 0.2 * (self.height_est - 2.0), 0.8, 2.0)
                        target_vx = -self.pid_vy.compute(err_y, dt, True, h_scale)
                        target_vy = self.pid_vx.compute(err_x, dt, True, h_scale)
                        max_speed = np.clip(0.8 + 0.3 * self.height_est, self.min_tracking_speed, self.max_tracking_speed)
                        self.cmd_vx = self.smooth_factor * self.cmd_vx + (1-self.smooth_factor) * np.clip(target_vx, -max_speed, max_speed)
                        self.cmd_vy = self.smooth_factor * self.cmd_vy + (1-self.smooth_factor) * np.clip(target_vy, -max_speed, max_speed)

                        if self.bottom_tag_corners is not None:
                            v_side = self.bottom_tag_corners[1] - self.bottom_tag_corners[0]
                            err_yaw = math.atan2(v_side[1], v_side[0])
                            t_yawspeed = np.clip(err_yaw * 0.8, -0.6, 0.6) if abs(err_yaw) > 0.05 else 0.0
                            self.cmd_yawspeed = 0.7 * self.cmd_yawspeed + 0.3 * t_yawspeed
                        else:
                            self.cmd_yawspeed = 0.0

                        pos_err = np.linalg.norm([err_x, err_y]) * self.height_est
                        yaw_err = abs(err_yaw) if self.bottom_tag_corners is not None else 0.0
                        if pos_err < self.drop_position_tolerance and yaw_err < self.drop_yaw_tolerance:
                            self.drop_ready_time += dt
                        else:
                            self.drop_ready_time = 0.0
                        if self.drop_ready_time >= self.drop_required_duration:
                            self.get_logger().info("✅ 投放！")
                            self.trigger_drop()
                            self.state = FlightState.DROP
                            self.drop_ready_time = 0.0
                    else:
                        if time_since_bottom > self.tag_timeout_bottom:
                            self.get_logger().warning("丢失目标 → SEARCHING")
                            self._reset_all_pid()
                            self.state = FlightState.SEARCHING
                        else:
                            self.cmd_vx *= 0.95; self.cmd_vy *= 0.95; self.cmd_yawspeed *= 0.9

                    twist = TwistStamped()
                    twist.header.stamp = self.get_clock().now().to_msg()
                    twist.header.frame_id = "base_link"
                    twist.twist.linear.x = self.cmd_vx
                    twist.twist.linear.y = self.cmd_vy
                    twist.twist.linear.z = self.cmd_vz
                    twist.twist.angular.z = self.cmd_yawspeed
                    self.cmd_vel_pub.publish(twist)
                    self.last_offboard_heartbeat = time.time()

            elif self.state == FlightState.DROP:
                twist = TwistStamped()
                twist.header.stamp = self.get_clock().now().to_msg()
                twist.header.frame_id = "base_link"
                twist.twist.linear.z = self.cmd_vz
                self.cmd_vel_pub.publish(twist)
                self.last_offboard_heartbeat = time.time()

        except Exception as e:
            self.get_logger().error(f"控制循环错误: {e}")

    def _reset_all_pid(self):
        for pid in [self.pid_vx, self.pid_vy, self.pid_yaw, self.app_pid_vx, self.app_pid_yaw, self.pid_alt]:
            pid.reset()

def main(args=None):
    rclpy.init(args=args)
    node = IntegratedDroneMission()
    executor = MultiThreadedExecutor(num_threads=6)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("⏹️ 手动停止")
    finally:
        node.camera_running = False
        time.sleep(0.5)
        if node.front_camera.isOpened(): node.front_camera.release()
        if node.bottom_camera.isOpened(): node.bottom_camera.release()
        node.destroy_node()
        executor.shutdown()
        rclpy.shutdown()

if __name__ == "__main__":
    main()