#!/usr/bin/env python3
"""
PX4 Offboard 搜索 + 逼近 + 抛物线投放 + 下视急停 + 下视稳定投放 + 自动返航 v1.6.2

新增（相对于 v1.6.1）：
- 下视摄像头稳定检测到目标 Tag 持续 1 秒后自动投放，与 v1.4.9 行为类似但保留原抛物线逻辑
- 其他所有逻辑（前视抛物线投放、滑行减速、下视急停）保持不变
"""

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode

import cv2
import numpy as np
import time
import threading
import math
import os
import queue
import shutil
from enum import Enum

import Jetson.GPIO as GPIO
import apriltag

try:
    from numba import cuda, uint8
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False

if HAS_CUDA:
    @cuda.jit
    def preprocess_kernel(input_frame, output_gray, gain, offset):
        x, y = cuda.grid(2)
        if x < output_gray.shape[1] and y < output_gray.shape[0]:
            idx = (y * output_gray.shape[1] + x) * 3
            b = input_frame[idx]; g = input_frame[idx+1]; r = input_frame[idx+2]
            gray = 0.299*r + 0.587*g + 0.114*b
            val = gray*gain + offset
            output_gray[y, x] = uint8(max(0.0, min(255.0, val)))


class FlightState(Enum):
    WAIT_CONNECTION = 0
    WAIT_STABILIZE = 1
    ARMING = 2
    TAKEOFF = 3
    SEARCHING = 4
    APPROACHING = 5
    LOST_COASTING = 6          # 前视丢失后的滑行减速状态
    LANDING = 7
    EMERGENCY_LAND = 8
    RETURN_TO_LAND = 9


class RobustSearchMission(Node):
    def __init__(self):
        super().__init__('robust_search_mission')

        # ---- 参数配置 ----
        self.declare_parameter('target_alt', 1.5)
        self.declare_parameter('takeoff_speed', 0.5)
        self.declare_parameter('search_yaw_rate', 20.0)
        self.declare_parameter('tag_id', 41)
        self.declare_parameter('tag_detect_threshold', 3)
        self.declare_parameter('heartbeat_timeout', 1.0)
        self.declare_parameter('state_timeout', 30.0)
        self.declare_parameter('max_takeoff_time', 60.0)
        self.declare_parameter('stabilize_wait', 3.0)
        self.declare_parameter('position_tolerance', 0.2)
        self.declare_parameter('camera_tilt_deg', 45.0)
        self.declare_parameter('lost_search_duration', 1.0)
        self.declare_parameter('yaw_rate_limit', 90.0)
        self.declare_parameter('contrast_gain', 1.2)
        self.declare_parameter('brightness_offset', 10.0)
        self.declare_parameter('lost_approach_wait_duration', 2.0)
        self.declare_parameter('landing_hold_duration', 3.0)
        self.declare_parameter('landing_position_tolerance', 0.1)
        self.declare_parameter('use_cuda', False)
        self.declare_parameter('altitude_jump_threshold', 0.5)
        self.declare_parameter('low_alt_threshold', 0.5)
        self.declare_parameter('alt_jump_count_threshold', 5)
        self.declare_parameter('use_hw_encoding', False)
        self.declare_parameter('vision_timeout', 3.0)
        self.declare_parameter('min_disk_space_mb', 500)
        self.declare_parameter('front_cam_sensor_id', 1)
        self.declare_parameter('bottom_cam_sensor_id', 0)
        self.declare_parameter('front_flip_method', 2)
        self.declare_parameter('bottom_flip_method', 0)
        self.declare_parameter('servo_pin', 32)
        self.declare_parameter('servo_close_duty', 2.5)
        self.declare_parameter('servo_open_duty', 7.5)
        self.declare_parameter('servo_hold_time', 1.0)
        self.declare_parameter('front_focal_length', 703.0)
        self.declare_parameter('bottom_focal_length', 722.0)
        self.declare_parameter('rtl_altitude', 1.5)
        self.declare_parameter('rtl_timeout', 30.0)

        # 逼近参数
        self.declare_parameter('approach_speed', 0.5)
        self.declare_parameter('align_duration', 2.0)
        self.declare_parameter('approach_lost_coast_decel', 0.5)   # 减速加速度 m/s^2
        self.declare_parameter('tag_size_m', 0.1865)               # Tag 边长 18.65cm
        self.declare_parameter('servo_delay_s', 0.3)               # 舵机延迟 0.3 秒
        self.declare_parameter('gravity', 9.8)

        # ----- 新增参数：下视稳定检测时间（秒）-----
        self.declare_parameter('bottom_stable_duration', 1.0)
        self.bottom_stable_duration = self.get_parameter('bottom_stable_duration').value

        # 读取参数
        self.target_alt = self.get_parameter('target_alt').value
        self.takeoff_speed = self.get_parameter('takeoff_speed').value
        self.search_yaw_rate_rad = math.radians(self.get_parameter('search_yaw_rate').value)
        self.target_tag_id = self.get_parameter('tag_id').value
        self.tag_detect_threshold = self.get_parameter('tag_detect_threshold').value
        self.heartbeat_timeout = self.get_parameter('heartbeat_timeout').value
        self.state_timeout = self.get_parameter('state_timeout').value
        self.max_takeoff_time = self.get_parameter('max_takeoff_time').value
        self.stabilize_wait = self.get_parameter('stabilize_wait').value
        self.position_tolerance = self.get_parameter('position_tolerance').value
        self.camera_tilt_deg = self.get_parameter('camera_tilt_deg').value
        self.lost_search_duration = self.get_parameter('lost_search_duration').value
        self.yaw_rate_limit = math.radians(self.get_parameter('yaw_rate_limit').value)
        self.contrast_gain = self.get_parameter('contrast_gain').value
        self.brightness_offset = self.get_parameter('brightness_offset').value
        self.lost_approach_wait_duration = self.get_parameter('lost_approach_wait_duration').value
        self.landing_hold_duration = self.get_parameter('landing_hold_duration').value
        self.landing_position_tolerance = self.get_parameter('landing_position_tolerance').value
        self.use_cuda = self.get_parameter('use_cuda').value and HAS_CUDA
        self.altitude_jump_threshold = self.get_parameter('altitude_jump_threshold').value
        self.low_alt_threshold = self.get_parameter('low_alt_threshold').value
        self.alt_jump_count_threshold = self.get_parameter('alt_jump_count_threshold').value
        self.use_hw_encoding = self.get_parameter('use_hw_encoding').value
        self.vision_timeout = self.get_parameter('vision_timeout').value
        self.min_disk_space_mb = self.get_parameter('min_disk_space_mb').value
        self.rtl_altitude = self.get_parameter('rtl_altitude').value
        self.rtl_timeout = self.get_parameter('rtl_timeout').value

        self.front_sensor_id = self.get_parameter('front_cam_sensor_id').value
        self.bottom_sensor_id = self.get_parameter('bottom_cam_sensor_id').value
        self.front_flip_method = self.get_parameter('front_flip_method').value
        self.bottom_flip_method = self.get_parameter('bottom_flip_method').value

        self.front_focal_length = self.get_parameter('front_focal_length').value
        self.bottom_focal_length = self.get_parameter('bottom_focal_length').value

        self.servo_pin = self.get_parameter('servo_pin').value
        self.servo_close_duty = self.get_parameter('servo_close_duty').value
        self.servo_open_duty = self.get_parameter('servo_open_duty').value
        self.servo_hold_time = self.get_parameter('servo_hold_time').value

        # 新参数
        self.approach_speed = self.get_parameter('approach_speed').value
        self.align_duration = self.get_parameter('align_duration').value
        self.approach_lost_coast_decel = self.get_parameter('approach_lost_coast_decel').value
        self.tag_size_m = self.get_parameter('tag_size_m').value
        self.servo_delay_s = self.get_parameter('servo_delay_s').value
        self.gravity = self.get_parameter('gravity').value

        # 警告
        if abs(self.front_focal_length - 703.0) < 1e-2:
            self.get_logger().warn("⚠️ 前视焦距仍为默认值，请根据实际标定修改！")
        if self.use_cuda:
            self.get_logger().warn("CUDA 预处理已启用，可能增加延迟")

        self.cos_tilt = math.cos(-math.radians(self.camera_tilt_deg))
        self.sin_tilt = math.sin(-math.radians(self.camera_tilt_deg))

        # 线程锁
        self.state_lock = threading.Lock()
        self.data_lock = threading.Lock()
        self.video_lock = threading.Lock()
        self.vehicle_state_lock = threading.Lock()
        self.pos_lock = threading.Lock()
        self.vel_lock = threading.Lock()
        self.detector_lock = threading.Lock()
        self.servo_lock = threading.Lock()

        # 状态变量
        self.flight_state = FlightState.WAIT_CONNECTION
        self.vehicle_state = State()
        self.current_pos = np.array([0.0, 0.0, 0.0])
        self.current_vel = np.array([0.0, 0.0, 0.0])   # ENU 速度
        self.current_yaw = 0.0
        self.takeoff_coords = [None, None]

        self.front_tag_detected = False
        self.front_tag_detect_counter = 0
        self.front_yaw_offset = 0.0
        self.front_pixel_offset = (0.0, 0.0)
        self.front_tag_corners = None      # 用于计算像素宽度

        self.bottom_tag_detected = False
        self.bottom_tag_detect_counter = 0
        self.bottom_pixel_offset = (0.0, 0.0)
        self._bottom_pose_time = 0.0

        # ----- 新增：下视稳定检测计时 -----
        self._bottom_detected_start_time = None   # 开始连续检测到的时刻
        self._bottom_detected_prev = False        # 上一次检测状态

        self._front_last_frame_time = 0.0
        self._bottom_last_frame_time = 0.0

        self._state_enter_time = self.get_clock().now()
        self._last_heartbeat = time.time()
        self._takeoff_start_z = 0.0
        self._takeoff_start_time = None
        self._desired_takeoff_z = None
        self._search_target_yaw = 0.0
        self._last_search_time = None
        self._mode_req_sent = False
        self._arm_req_sent = False
        self._mode_retry_count = {}
        self._mode_retry_time = {}

        # 前视逼近子状态
        self._approach_substate = 0          # 0:偏航对齐, 1:前进接近
        self._approach_align_start = None
        self._approach_smooth_xy = None
        self._approach_lost_time = None

        # 滑行减速相关
        self._coast_vel_setpoint = 0.0           # 当前期望前向速度（逐渐减小）
        self._last_known_distance = None         # 进入滑行前最后估算的距离
        self._last_known_forward_speed = None    # 进入滑行前最后记录的速度
        self._coast_start_time = None            # 滑行开始时间（用于盲投）

        self.servo_triggered = False
        self.servo_action_complete = False
        self.servo_event = threading.Event()
        self.servo_pwm = None
        self._servo_trigger_in_progress = False
        self._servo_shutdown = False              # 通知舵机线程退出

        self._last_published_yaw = None
        self._last_yaw_time = None

        self.video_queue = queue.Queue(maxsize=100)
        self._video_dropped = 0
        self._last_video_log_time = time.time()
        self._last_disk_check_time = time.time()
        self.video_writer_thread = threading.Thread(target=self.video_writer_loop, daemon=True)
        self.video_writer_thread.start()
        self.recording = False
        self.front_video_writer = None
        self.bottom_video_writer = None

        self._last_mode_req_time = {}
        self._last_altitude = None
        self._alt_jump_counter = 0
        self._rtl_start_time = None
        self._arming_step = None
        self._arming_enter_time = None

        self.init_servo()

        # ROS 接口
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE,
                         history=HistoryPolicy.KEEP_LAST, depth=5)
        self.state_sub = self.create_subscription(State, '/mavros/state', self.state_cb, qos)
        self.pos_sub = self.create_subscription(PoseStamped, '/mavros/local_position/pose', self.pos_cb, qos)
        self.vel_sub = self.create_subscription(TwistStamped, '/mavros/local_position/velocity_local', self.vel_cb, qos)
        self.local_pos_pub = self.create_publisher(PoseStamped, '/mavros/setpoint_position/local', 10)
        self.set_mode_cli = self.create_client(SetMode, '/mavros/set_mode')
        self.arm_cli = self.create_client(CommandBool, '/mavros/cmd/arming')

        self.init_cameras()
        options = apriltag.DetectorOptions(families="tag36h11")
        options.quad_decimate = 1.0
        self.detector_front = apriltag.Detector(options)
        self.detector_bottom = apriltag.Detector(options)

        self.control_timer = self.create_timer(0.05, self.control_loop)
        self.watchdog_timer = self.create_timer(0.5, self.watchdog_check)
        self.cam_front_thread = threading.Thread(target=self.front_camera_loop, daemon=True)
        self.cam_bottom_thread = threading.Thread(target=self.bottom_camera_loop, daemon=True)
        self.cam_front_thread.start()
        self.cam_bottom_thread.start()

        self.get_logger().info("🚀 v1.6.2：新增下视稳定2秒投放，保留抛物线投放")
        self.get_logger().info(f"🕹️ 舵机已归位（引脚 {self.servo_pin}，占空比 {self.servo_close_duty}）")

    # ========== 舵机 ==========
    def init_servo(self):
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(self.servo_pin, GPIO.OUT)
        self.servo_pwm = GPIO.PWM(self.servo_pin, 50)
        self.servo_pwm.start(self.servo_close_duty)
        time.sleep(0.5)
        self.get_logger().info("✅ 舵机归位完成")

    def _servo_action_thread(self):
        with self.servo_lock:
            self._servo_trigger_in_progress = True
            try:
                self.get_logger().info("🎁 触发投放：打开舵机")
                self.servo_pwm.ChangeDutyCycle(self.servo_open_duty)
                # 分段睡眠，每0.1秒检查一次退出标志，避免阻塞销毁
                for _ in range(int(self.servo_hold_time * 10)):
                    if self._servo_shutdown:
                        break
                    time.sleep(0.1)
                self.get_logger().info("🎁 投放完成：关闭舵机")
                if not self._servo_shutdown:
                    self.servo_pwm.ChangeDutyCycle(self.servo_close_duty)
                time.sleep(0.2)
                self.servo_action_complete = True
            except Exception as e:
                self.get_logger().error(f"舵机异常: {e}")
                self.servo_action_complete = True
            finally:
                self.servo_event.set()
                self._servo_trigger_in_progress = False

    def trigger_servo_drop(self):
        if self.servo_triggered or self._servo_trigger_in_progress:
            return
        self.servo_triggered = True
        self.servo_action_complete = False
        self.servo_event.clear()
        self.get_logger().info("🚁 开始投放...")
        threading.Thread(target=self._servo_action_thread, daemon=True).start()

    # ========== 访问器 ==========
    def get_mode(self):
        with self.vehicle_state_lock: return self.vehicle_state.mode
    def is_armed(self):
        with self.vehicle_state_lock: return self.vehicle_state.armed
    def is_connected(self):
        with self.vehicle_state_lock: return self.vehicle_state.connected
    def get_state(self):
        with self.state_lock: return self.flight_state
    def set_state(self, new_state):
        with self.state_lock: self.flight_state = new_state

    # ========== 回调 ==========
    def state_cb(self, msg):
        with self.vehicle_state_lock: self.vehicle_state = msg
    def pos_cb(self, msg):
        pos = msg.pose.position
        q = msg.pose.orientation
        yaw = math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))
        with self.pos_lock:
            self.current_pos = np.array([pos.x, pos.y, pos.z])
            self.current_yaw = yaw
    def vel_cb(self, msg):
        with self.vel_lock:
            self.current_vel = np.array([msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z])

    # ========== 摄像头初始化 ==========
    def init_cameras(self):
        gst_front = (f'nvarguscamerasrc sensor-id={self.front_sensor_id} ! '
                     'video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! '
                     f'nvvidconv flip-method={self.front_flip_method} ! '
                     'video/x-raw, width=640, height=480, format=BGRx ! '
                     'videoconvert ! video/x-raw, format=BGR ! appsink drop=True max-buffers=1')
        self.cap_front = cv2.VideoCapture(gst_front, cv2.CAP_GSTREAMER)
        if not self.cap_front.isOpened():
            self.get_logger().warn("前视 CSI 失败，尝试 USB 0")
            self.cap_front = cv2.VideoCapture(0)
            self.cap_front.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap_front.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        gst_bottom = (f'nvarguscamerasrc sensor-id={self.bottom_sensor_id} ! '
                      'video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! '
                      f'nvvidconv flip-method={self.bottom_flip_method} ! '
                      'video/x-raw, width=640, height=480, format=BGRx ! '
                      'videoconvert ! video/x-raw, format=BGR ! appsink drop=True max-buffers=1')
        self.cap_bottom = cv2.VideoCapture(gst_bottom, cv2.CAP_GSTREAMER)
        if not self.cap_bottom.isOpened():
            self.get_logger().warn("下视 CSI 失败，尝试 USB 1")
            self.cap_bottom = cv2.VideoCapture(1)
            self.cap_bottom.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap_bottom.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # ========== 视频写入 ==========
    def video_writer_loop(self):
        while rclpy.ok():
            try:
                item = self.video_queue.get(timeout=0.5)
                if item is None: break
                cam_id, frame = item
                with self.video_lock:
                    if cam_id == 'front' and self.front_video_writer:
                        self.front_video_writer.write(frame)
                    elif cam_id == 'bottom' and self.bottom_video_writer:
                        self.bottom_video_writer.write(frame)
            except queue.Empty:
                pass
            if self.recording and time.time() - self._last_disk_check_time > 10.0:
                self._check_disk_space()
                self._last_disk_check_time = time.time()

    def _check_disk_space(self):
        try:
            save_dir = os.path.expanduser("~/桌面")
            if not os.path.exists(save_dir):
                save_dir = os.path.expanduser("~/Desktop")
            stat = shutil.disk_usage(save_dir)
            free_mb = stat.free / (1024*1024)
            if free_mb < self.min_disk_space_mb:
                self.get_logger().error(f"磁盘不足 {free_mb:.0f} MB，停止录制")
                self.stop_video()
        except Exception as e:
            self.get_logger().warn(f"磁盘检查失败: {e}")

    # ========== 摄像头采集线程 ==========
    def front_camera_loop(self):
        d_frame = d_gray = None
        if self.use_cuda:
            d_frame = cuda.device_array((480*640*3,), dtype=np.uint8)
            d_gray = cuda.device_array((480,640), dtype=np.uint8)
        adaptive_gain = self.contrast_gain
        lost_frames = 0
        while rclpy.ok():
            ret, frame = self.cap_front.read()
            if not ret: continue
            with self.data_lock:
                self._front_last_frame_time = time.time()
            detections, yaw_off, pix_off, corners = self.detect_tag(frame, True, d_frame, d_gray, adaptive_gain)
            detected = False
            for d in detections:
                if d.tag_id == self.target_tag_id:
                    detected = True
                    cv2.polylines(frame, [np.array(d.corners, dtype=np.int32)], True, (0,255,0), 2)
                    break
            if detected:
                lost_frames = 0
                adaptive_gain = self.contrast_gain
            else:
                lost_frames += 1
                if lost_frames > 10:
                    adaptive_gain = [1.0,1.5,2.0][(lost_frames//10)%3]
            try:
                self.video_queue.put(('front', frame), block=False)
            except queue.Full:
                self._video_dropped += 1
            with self.data_lock:
                if yaw_off is not None:
                    self.front_tag_detect_counter = min(10, self.front_tag_detect_counter+1)
                else:
                    self.front_tag_detect_counter = max(0, self.front_tag_detect_counter-1)
                self.front_tag_detected = self.front_tag_detect_counter >= self.tag_detect_threshold
                self.front_yaw_offset = yaw_off if yaw_off is not None else 0.0
                self.front_pixel_offset = pix_off if pix_off is not None else (0.0, 0.0)
                self.front_tag_corners = corners

    def bottom_camera_loop(self):
        d_frame = d_gray = None
        if self.use_cuda:
            d_frame = cuda.device_array((480*640*3,), dtype=np.uint8)
            d_gray = cuda.device_array((480,640), dtype=np.uint8)
        adaptive_gain = self.contrast_gain
        lost_frames = 0
        while rclpy.ok():
            ret, frame = self.cap_bottom.read()
            if not ret: continue
            with self.data_lock:
                self._bottom_last_frame_time = time.time()
            detections, yaw_off, pix_off, _ = self.detect_tag(frame, False, d_frame, d_gray, adaptive_gain)
            detected = False
            for d in detections:
                if d.tag_id == self.target_tag_id:
                    detected = True
                    cv2.polylines(frame, [np.array(d.corners, dtype=np.int32)], True, (0,255,0), 2)
                    break
            if detected:
                lost_frames = 0
                adaptive_gain = self.contrast_gain
            else:
                lost_frames += 1
                if lost_frames > 10:
                    adaptive_gain = [1.0,1.5,2.0][(lost_frames//10)%3]
            try:
                self.video_queue.put(('bottom', frame), block=False)
            except queue.Full:
                self._video_dropped += 1
            with self.data_lock:
                if yaw_off is not None:
                    self.bottom_tag_detect_counter = min(10, self.bottom_tag_detect_counter+1)
                else:
                    self.bottom_tag_detect_counter = max(0, self.bottom_tag_detect_counter-1)
                self.bottom_tag_detected = self.bottom_tag_detect_counter >= self.tag_detect_threshold
                self.bottom_pixel_offset = pix_off if pix_off is not None else (0.0, 0.0)
                self._bottom_pose_time = time.time()

    def detect_tag(self, frame, is_front, d_frame, d_gray, gain):
        if self.use_cuda and d_frame is not None:
            d_frame.copy_to_device(frame.ravel())
            tpb = (16,16); bpg = (math.ceil(640/16), math.ceil(480/16))
            preprocess_kernel[bpg, tpb](d_frame, d_gray, gain, self.brightness_offset)
            gray = d_gray.copy_to_host()
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        with self.detector_lock:
            detections = (self.detector_front if is_front else self.detector_bottom).detect(gray)
        yaw_off = None
        pixel_off = None
        corners = None
        focal = self.front_focal_length if is_front else self.bottom_focal_length
        for d in detections:
            if d.tag_id == self.target_tag_id:
                dx = d.center[0] - 320.0
                dy = d.center[1] - 240.0
                corners = d.corners
                if is_front:
                    xn = dx / focal
                    yn = dy / focal
                    yaw_off = math.atan2(-xn, yn*self.sin_tilt + self.cos_tilt)
                    pixel_off = (dx, dy)
                else:
                    yaw_off = -math.atan2(dx, -dy)   # 下视偏航（备用，但不使用）
                    pixel_off = (dx, dy)
                break
        return detections, yaw_off, pixel_off, corners

    # ========== 辅助函数：计算距离和投放时机 ==========
    def estimate_distance_to_tag(self, corners):
        """通过像素宽度估算斜距（米）"""
        if corners is None or len(corners) < 4:
            return None
        # 取上边宽度作为参考
        w_px = np.linalg.norm(corners[1] - corners[0])
        if w_px < 1:
            return None
        dist = (self.tag_size_m * self.front_focal_length) / w_px
        # 考虑摄像头俯仰角，斜距转水平距离（近似）
        horizontal_dist = dist * math.cos(math.radians(self.camera_tilt_deg))
        return horizontal_dist

    def compute_forward_velocity(self):
        """计算机头方向的速度分量（正值向前）"""
        with self.vel_lock:
            vx = self.current_vel[0]
            vy = self.current_vel[1]
        yaw = self.current_yaw
        forward = vx * math.cos(yaw) + vy * math.sin(yaw)
        return forward

    def should_drop_parabolic(self, distance, forward_speed, height):
        """判断是否到达抛物线投放点"""
        if distance is None or forward_speed <= 0.01 or height <= 0.2:
            return False
        t_fall = math.sqrt(2.0 * height / self.gravity)
        drop_dist = forward_speed * (t_fall + self.servo_delay_s)
        # 留一点余量，避免投放过晚
        if distance <= drop_dist + 0.05:
            self.get_logger().info(f"🎯 抛物线触发: dist={distance:.2f}, drop_dist={drop_dist:.2f}, v={forward_speed:.2f}, H={height:.2f}")
            return True
        return False

    # ========== 主控制循环 ==========
    def control_loop(self):
        try:
            with self.pos_lock:
                curr_pos = self.current_pos.copy()
                curr_yaw = self.current_yaw
                alt = curr_pos[2]
            if any(math.isnan(v) for v in curr_pos):
                return

            # 高度跳变检测
            if self._last_altitude is not None:
                delta_alt = alt - self._last_altitude
                if delta_alt < -self.altitude_jump_threshold:
                    self._alt_jump_counter += 1
                    if self._alt_jump_counter >= self.alt_jump_count_threshold:
                        self.get_logger().error(f"高度快速下降 {self._alt_jump_counter} 次，紧急降落")
                        self.trigger_emergency_land()
                        self._last_altitude = alt
                        self._alt_jump_counter = 0
                        return
                else:
                    self._alt_jump_counter = 0
            self._last_altitude = alt

            now = self.get_clock().now()
            current_state = self.get_state()
            if not hasattr(self, '_last_control_time'):
                self._last_control_time = now
            dt = (now - self._last_control_time).nanoseconds / 1e9
            self._last_control_time = now
            dt = max(0.01, min(dt, 0.1))

            # 紧急降落/降落状态
            if current_state in (FlightState.EMERGENCY_LAND, FlightState.LANDING):
                self.publish_setpoint(curr_pos, curr_pos[:2], curr_yaw, None)
                if self.get_mode() != 'AUTO.LAND':
                    self.request_mode('AUTO.LAND')
                if current_state == FlightState.EMERGENCY_LAND:
                    elapsed = (now - self._state_enter_time).nanoseconds / 1e9
                    if elapsed > 15.0 and self.is_armed():
                        self.get_logger().error("紧急降落超时，强制 disarm")
                        self.request_arm(False)
                self._last_heartbeat = time.time()
                return

            target_yaw = curr_yaw
            desired_xy = np.array([curr_pos[0], curr_pos[1]])
            z_desired = self.target_alt

            # ========== 状态机 ==========
            if current_state == FlightState.WAIT_CONNECTION:
                if self.is_connected():
                    self._enter_state(FlightState.WAIT_STABILIZE)

            elif current_state == FlightState.WAIT_STABILIZE:
                self.publish_setpoint(curr_pos, desired_xy, curr_yaw, alt)
                if (now - self._state_enter_time).nanoseconds / 1e9 > self.stabilize_wait:
                    self.takeoff_coords = [curr_pos[0], curr_pos[1]]
                    self._enter_state(FlightState.ARMING)

            elif current_state == FlightState.ARMING:
                self.publish_setpoint(curr_pos, desired_xy, curr_yaw, alt)
                if self._arming_step is None:
                    self._arming_step = 0
                    self._arming_enter_time = now

                if self._arming_step == 0:
                    if self.get_mode() != 'OFFBOARD':
                        self.request_mode('OFFBOARD')
                    elapsed_mode = (now - self._arming_enter_time).nanoseconds / 1e9
                    if self.get_mode() == 'OFFBOARD' or elapsed_mode > 2.0:
                        self._arming_step = 1
                        self.get_logger().info(f"OFFBOARD 模式确认, 进入解锁步骤")
                elif self._arming_step == 1:
                    if not self.is_armed():
                        self.request_arm(True)
                    elapsed_arm = (now - self._arming_enter_time).nanoseconds / 1e9
                    if self.is_armed() or elapsed_arm > 3.0:
                        if not self.is_armed():
                            self.get_logger().warn("解锁未确认，强制继续")
                        self._arming_step = 2
                elif self._arming_step == 2:
                    self.start_video()
                    self._enter_state(FlightState.TAKEOFF)

            elif current_state == FlightState.TAKEOFF:
                if self._takeoff_start_time is None:
                    self._takeoff_start_z = curr_pos[2]
                    self._takeoff_start_time = now
                elapsed = (now - self._takeoff_start_time).nanoseconds / 1e9
                self._desired_takeoff_z = min(self._takeoff_start_z + self.takeoff_speed * elapsed, self.target_alt)
                z_desired = self._desired_takeoff_z
                if abs(curr_pos[2] - self.target_alt) < self.position_tolerance:
                    self.get_logger().info("✅ 到达目标高度，进入搜索")
                    self._enter_state(FlightState.SEARCHING)
                    self._search_target_yaw = curr_yaw
                    self._last_search_time = now

            elif current_state == FlightState.SEARCHING:
                with self.data_lock:
                    front_det = self.front_tag_detected
                if front_det:
                    self._enter_state(FlightState.APPROACHING)
                else:
                    if self.takeoff_coords[0] is not None:
                        desired_xy = np.array([self.takeoff_coords[0], self.takeoff_coords[1]])
                    if self._last_search_time is None:
                        self._search_target_yaw = curr_yaw
                        self._last_search_time = now
                    dt_s = (now - self._last_search_time).nanoseconds / 1e9
                    self._search_target_yaw += self.search_yaw_rate_rad * dt_s
                    self._search_target_yaw %= 2*math.pi
                    self._last_search_time = now
                    target_yaw = self._search_target_yaw

            # ========== 前视逼近（偏航对齐 + 匀速前进 + 抛物线投放）==========
            elif current_state == FlightState.APPROACHING:
                with self.data_lock:
                    front_det = self.front_tag_detected
                    front_yaw_off = self.front_yaw_offset
                    dx_px, dy_px = self.front_pixel_offset
                    corners = self.front_tag_corners

                # 子状态 0：偏航对齐（原地旋转）
                if self._approach_substate == 0:
                    target_yaw = curr_yaw + front_yaw_off
                    desired_xy = curr_pos[:2]   # 保持当前位置
                    z_desired = self.target_alt
                    if abs(front_yaw_off) < 0.05:   # 偏航误差 < 0.05 rad
                        if self._approach_align_start is None:
                            self._approach_align_start = now
                        else:
                            align_dur = (now - self._approach_align_start).nanoseconds / 1e9
                            if align_dur >= self.align_duration:
                                self.get_logger().info("✅ 前视偏航对齐完成，开始前进接近")
                                self._approach_substate = 1
                                self._approach_smooth_xy = curr_pos[:2].copy()
                                self._approach_align_start = None
                    else:
                        self._approach_align_start = None

                # 子状态 1：前进接近（匀速，高度不变，实时计算投放时机）
                elif self._approach_substate == 1:
                    if front_det:
                        self._approach_lost_time = None
                        # 计算目标点：沿机头方向匀速移动
                        forward_dir = np.array([math.cos(curr_yaw), math.sin(curr_yaw)])
                        if self._approach_smooth_xy is None:
                            self._approach_smooth_xy = curr_pos[:2].copy()
                        self._approach_smooth_xy += forward_dir * (self.approach_speed * dt)
                        desired_xy = self._approach_smooth_xy
                        z_desired = self.target_alt
                        target_yaw = curr_yaw + front_yaw_off   # 持续修正偏航

                        # 抛物线投放判断
                        if not self.servo_triggered and not self._servo_trigger_in_progress:
                            distance = self.estimate_distance_to_tag(corners)
                            forward_speed = self.compute_forward_velocity()
                            if distance is not None and forward_speed > 0.01:
                                if self.should_drop_parabolic(distance, forward_speed, alt):
                                    self.trigger_servo_drop()
                                    self._enter_state(FlightState.RETURN_TO_LAND)
                                    return
                    else:
                        # 前视丢失：保存最后信息并进入滑行减速状态
                        if self._approach_lost_time is None:
                            self._approach_lost_time = now
                            self.get_logger().warn("前视目标丢失，进入滑行减速")
                            # 保存最后已知的距离和速度（用于滑行中继续判断）
                            self._last_known_distance = self.estimate_distance_to_tag(corners) if corners else None
                            self._last_known_forward_speed = self.compute_forward_velocity()
                            self._coast_vel_setpoint = self.approach_speed
                            self._coast_start_time = now
                            self._enter_state(FlightState.LOST_COASTING)
                            return

            # ========== 前视丢失后的滑行减速（增加抛物线投放判断）==========
            elif current_state == FlightState.LOST_COASTING:
                # 减速
                self._coast_vel_setpoint -= self.approach_lost_coast_decel * dt
                if self._coast_vel_setpoint < 0.0:
                    self._coast_vel_setpoint = 0.0
                # 沿当前机头方向继续移动（速度递减）
                forward_dir = np.array([math.cos(curr_yaw), math.sin(curr_yaw)])
                if self._approach_smooth_xy is None:
                    self._approach_smooth_xy = curr_pos[:2].copy()
                self._approach_smooth_xy += forward_dir * (self._coast_vel_setpoint * dt)
                desired_xy = self._approach_smooth_xy
                z_desired = self.target_alt
                target_yaw = curr_yaw   # 不再修正偏航

                # 1. 检查下视是否看到 Tag（急停返航）
                with self.data_lock:
                    bottom_det = self.bottom_tag_detected
                if bottom_det:
                    self.get_logger().info("🛑 下视检测到 Tag，立即急停返航")
                    desired_xy = curr_pos[:2]
                    self._enter_state(FlightState.RETURN_TO_LAND)
                    return

                # 2. 滑行中继续尝试抛物线投放（基于最后已知距离和速度估算当前剩余距离）
                if not self.servo_triggered and not self._servo_trigger_in_progress:
                    # 估算已滑行距离
                    coasted_dist = (self.approach_speed - self._coast_vel_setpoint) / self.approach_lost_coast_decel * self.approach_speed * 0.5  # 简化梯形积分
                    # 更精确：累加每帧滑行位移
                    if not hasattr(self, '_coasted_total'):
                        self._coasted_total = 0.0
                    self._coasted_total += self._coast_vel_setpoint * dt
                    remaining_dist = (self._last_known_distance - self._coasted_total) if self._last_known_distance is not None else None
                    forward_speed = self._coast_vel_setpoint  # 当前滑行速度
                    if remaining_dist is not None and forward_speed > 0.01:
                        if self.should_drop_parabolic(remaining_dist, forward_speed, alt):
                            self.trigger_servo_drop()
                            self._enter_state(FlightState.RETURN_TO_LAND)
                            return
                    # 强制盲投：滑行超过 1.5 秒且仍未投放，认为已足够接近，立即投放
                    if self._coast_start_time is not None:
                        coast_duration = (now - self._coast_start_time).nanoseconds / 1e9
                        if coast_duration > 1.5:
                            self.get_logger().info("⏰ 滑行超时，强制投放")
                            self.trigger_servo_drop()
                            self._enter_state(FlightState.RETURN_TO_LAND)
                            return

                # 如果速度已经降为零，且仍未投放/下视，则返回搜索模式
                if self._coast_vel_setpoint <= 0.01:
                    self.get_logger().warn("滑行减速至零，未检测到下视，返回搜索")
                    self.takeoff_coords = [curr_pos[0], curr_pos[1]]
                    self._enter_state(FlightState.SEARCHING)
                    return

            # ========== 返航降落（投放后或下视急停后）==========
            elif current_state == FlightState.RETURN_TO_LAND:
                if self._rtl_start_time is None:
                    self._rtl_start_time = now
                    self.request_mode('AUTO.RTL')
                    self.get_logger().info("✈️ 已请求 AUTO.RTL，无人机将自动返航降落")
                elapsed_rtl = (now - self._rtl_start_time).nanoseconds / 1e9
                if elapsed_rtl > self.rtl_timeout:
                    self.get_logger().error(f"返航超时 {elapsed_rtl:.1f}s，紧急降落")
                    self.trigger_emergency_land()
                if not self.is_armed() or alt < 0.1:
                    self.get_logger().info("🛬 已着陆，任务结束")
                    self.stop_video()
                    self._enter_state(FlightState.LANDING)
                self.publish_setpoint(curr_pos, desired_xy, target_yaw, alt)
                self._last_heartbeat = time.time()
                return

            # ========== 新增：下视稳定检测投放（2秒） ==========
            # 在未投放、未触发舵机、且非滑行状态（避免与急停冲突）下，检测下视是否稳定
            if (not self.servo_triggered and not self._servo_trigger_in_progress and
                current_state != FlightState.LOST_COASTING and current_state != FlightState.RETURN_TO_LAND):
                with self.data_lock:
                    bottom_det = self.bottom_tag_detected
                now_time = time.time()
                if bottom_det:
                    if not self._bottom_detected_prev:
                        # 刚开始检测到
                        self._bottom_detected_start_time = now_time
                        self._bottom_detected_prev = True
                    else:
                        # 持续检测中
                        if self._bottom_detected_start_time is not None:
                            duration = now_time - self._bottom_detected_start_time
                            if duration >= self.bottom_stable_duration:
                                self.get_logger().info(f"✅ 下视稳定检测到 Tag 持续 {duration:.1f} 秒，触发投放")
                                self.trigger_servo_drop()
                                self._enter_state(FlightState.RETURN_TO_LAND)
                                # 投放后直接跳出本次控制循环，避免重复发布
                                self._last_heartbeat = time.time()
                                return
                else:
                    # 未检测到，重置计时
                    self._bottom_detected_start_time = None
                    self._bottom_detected_prev = False

            # 偏航速率限制
            if self._last_published_yaw is not None and self._last_yaw_time is not None:
                yaw_diff = target_yaw - self._last_published_yaw
                yaw_diff = math.atan2(math.sin(yaw_diff), math.cos(yaw_diff))
                max_delta = self.yaw_rate_limit * dt
                if abs(yaw_diff) > max_delta:
                    target_yaw = self._last_published_yaw + math.copysign(max_delta, yaw_diff)
            self._last_published_yaw = target_yaw
            self._last_yaw_time = now

            self.publish_setpoint(curr_pos, desired_xy, target_yaw, z_desired)
            self._last_heartbeat = time.time()

        except Exception as e:
            self.get_logger().error(f"❌ 控制循环异常: {e}", throttle_duration_sec=1.0)

    # ========== 安全监控 ==========
    def watchdog_check(self):
        now = self.get_clock().now()
        if time.time() - self._last_heartbeat > self.heartbeat_timeout:
            self.get_logger().error("控制循环丢失，紧急降落")
            self.trigger_emergency_land()
            return
        with self.data_lock:
            front_time = self._front_last_frame_time
            bottom_time = self._bottom_last_frame_time
        vision_now = time.time()
        if front_time > 0 and vision_now - front_time > self.vision_timeout:
            self.get_logger().error("前视摄像头无新帧，紧急降落")
            self.trigger_emergency_land()
            return
        if bottom_time > 0 and vision_now - bottom_time > self.vision_timeout:
            self.get_logger().error("下视摄像头无新帧，紧急降落")
            self.trigger_emergency_land()
            return
        elapsed = (now - self._state_enter_time).nanoseconds / 1e9
        state = self.get_state()
        if state in (FlightState.LOST_COASTING, FlightState.EMERGENCY_LAND, FlightState.LANDING, FlightState.RETURN_TO_LAND, FlightState.ARMING):
            return
        if state == FlightState.TAKEOFF:
            if elapsed > self.max_takeoff_time:
                self.get_logger().error(f"TAKEOFF 超时，紧急降落")
                self.trigger_emergency_land()
            return
        timeout = self.state_timeout * (2 if state == FlightState.SEARCHING else 1)
        if elapsed > timeout:
            self.get_logger().error(f"状态 {state.name} 超时 ({elapsed:.1f}s)，触发降落")
            self.trigger_emergency_land()

    def trigger_emergency_land(self):
        if self.get_state() == FlightState.EMERGENCY_LAND:
            return
        self.get_logger().error("🚨 切换至紧急降落")
        self._enter_state(FlightState.EMERGENCY_LAND)
        self.request_mode('AUTO.LAND')
        self.stop_video()

    def _enter_state(self, new_state):
        with self.state_lock:
            if self.flight_state == new_state:
                return
            self.get_logger().info(f"[状态机] {self.flight_state.name} → {new_state.name}")
            self.flight_state = new_state
            self._state_enter_time = self.get_clock().now()
            self._mode_req_sent = False
            self._arm_req_sent = False
            if new_state != FlightState.TAKEOFF:
                self._desired_takeoff_z = None
                self._takeoff_start_time = None
            if new_state == FlightState.SEARCHING:
                self._last_search_time = None
                self._approach_substate = 0
                self._approach_align_start = None
                self._approach_smooth_xy = None
                self._approach_lost_time = None
                self.servo_triggered = False
                self.servo_action_complete = False
                self._servo_trigger_in_progress = False
                self._coasted_total = 0.0   # 重置滑行累计距离
                # 重置下视稳定检测计时
                self._bottom_detected_start_time = None
                self._bottom_detected_prev = False
            elif new_state == FlightState.APPROACHING:
                self._approach_substate = 0
                self._approach_align_start = None
                self._approach_lost_time = None
                self._approach_smooth_xy = None
                with self.pos_lock:
                    self._approach_smooth_xy = np.array([self.current_pos[0], self.current_pos[1]])
                # 重置下视稳定检测计时
                self._bottom_detected_start_time = None
                self._bottom_detected_prev = False
            elif new_state == FlightState.LOST_COASTING:
                self._coast_vel_setpoint = self.approach_speed
                self._coast_start_time = self.get_clock().now()
                self._coasted_total = 0.0
            elif new_state == FlightState.RETURN_TO_LAND:
                self._rtl_start_time = None
            elif new_state == FlightState.LANDING:
                self.request_mode('AUTO.LAND')
                self.stop_video()
            elif new_state == FlightState.WAIT_CONNECTION:
                self.servo_triggered = False
                self.servo_action_complete = False
                self._servo_trigger_in_progress = False
            if new_state == FlightState.ARMING:
                self._arming_step = None
                self._arming_enter_time = None

    # ========== 模式/解锁 ==========
    def request_mode(self, target_mode):
        if self.get_mode() == target_mode:
            return
        now = time.time()
        retry_key = target_mode
        if retry_key not in self._mode_retry_count:
            self._mode_retry_count[retry_key] = 0
        if self._mode_retry_count[retry_key] >= 3:
            self.get_logger().error(f"模式 {target_mode} 重试超限，紧急降落")
            self.trigger_emergency_land()
            return
        min_interval = 2 ** self._mode_retry_count[retry_key]
        last_req = self._last_mode_req_time.get(target_mode, 0)
        if now - last_req < min_interval:
            return
        self._last_mode_req_time[target_mode] = now
        self.get_logger().info(f"→ 请求模式: {target_mode} (重试 #{self._mode_retry_count[retry_key]})")
        req = SetMode.Request()
        req.custom_mode = target_mode
        future = self.set_mode_cli.call_async(req)
        future.add_done_callback(lambda f, m=target_mode: self._mode_response_callback(f, m))

    def _mode_response_callback(self, future, target_mode):
        try:
            result = future.result()
            if result and result.mode_sent:
                self.get_logger().info(f"✅ {target_mode} 已接受")
                self._mode_retry_count[target_mode] = 0
            else:
                if self.get_mode() == target_mode:
                    self.get_logger().info(f"⚠️ {target_mode} 实际已进入")
                    self._mode_retry_count[target_mode] = 0
                else:
                    self.get_logger().error(f"❌ 模式切换失败，将重试")
                    self._mode_retry_count[target_mode] += 1
        except Exception as e:
            self.get_logger().error(f"模式服务异常: {e}")
            self._mode_retry_count[target_mode] += 1

    def request_arm(self, value):
        if self._arm_req_sent:
            return
        req = CommandBool.Request()
        req.value = value
        future = self.arm_cli.call_async(req)
        future.add_done_callback(lambda f, v=value: self._arm_response_callback(f, v))
        self._arm_req_sent = True

    def _arm_response_callback(self, future, value):
        try:
            result = future.result()
            if result and result.success:
                self.get_logger().info("✅ 解锁成功")
                self._arm_req_sent = False
            else:
                if self.is_armed() == bool(value):
                    self.get_logger().info("⚠️ 状态匹配")
                else:
                    self.get_logger().error("❌ 解锁失败")
                    self._arm_req_sent = False
        except Exception as e:
            self.get_logger().error(f"解锁异常: {e}")
            self._arm_req_sent = False

    def publish_setpoint(self, curr_pos, desired_xy, yaw, z_des):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.pose.position.x = float(desired_xy[0])
        msg.pose.position.y = float(desired_xy[1])
        if z_des is not None:
            msg.pose.position.z = float(z_des)
        else:
            st = self.get_state()
            if st == FlightState.TAKEOFF and self._desired_takeoff_z is not None:
                msg.pose.position.z = self._desired_takeoff_z
            elif st in (FlightState.SEARCHING, FlightState.APPROACHING, FlightState.LOST_COASTING):
                msg.pose.position.z = self.target_alt
            else:
                msg.pose.position.z = float(curr_pos[2])
        msg.pose.orientation.z = math.sin(yaw/2)
        msg.pose.orientation.w = math.cos(yaw/2)
        self.local_pos_pub.publish(msg)

    # ========== 视频录制（修复 Jetson 硬编）==========
    def start_video(self):
        with self.video_lock:
            if self.recording:
                return
            save_dir = os.path.expanduser("~/桌面")
            if not os.path.exists(save_dir):
                save_dir = os.path.expanduser("~/Desktop")
            try:
                stat = shutil.disk_usage(save_dir)
                if stat.free / (1024 * 1024) < self.min_disk_space_mb:
                    self.get_logger().warn(f"磁盘不足，不录制")
                    return
            except Exception as e:
                self.get_logger().warn(f"磁盘检查失败: {e}")
            os.makedirs(save_dir, exist_ok=True)
            ts = int(time.time())
            front_path = os.path.join(save_dir, f"front_{ts}.mp4")
            bottom_path = os.path.join(save_dir, f"bottom_{ts}.mp4")
            def gst_pipe(path):
                if self.use_hw_encoding:
                    # 修复：使用 Jetson 正确的硬编插件 nvv4l2h264enc，码率单位 bps -> 3000000 = 3Mbps
                    return (f"appsrc ! videoconvert ! video/x-raw,format=I420 ! "
                            f"nvvidconv ! nvv4l2h264enc bitrate=3000000 ! "
                            f"h264parse ! mp4mux fragment-duration=100 ! filesink location={path}")
                else:
                    # 软编码保持原样，bitrate 单位 kbps
                    return (f"appsrc ! videoconvert ! video/x-raw,format=I420 ! "
                            f"x264enc tune=zerolatency bitrate=3000 speed-preset=ultrafast ! "
                            f"h264parse ! mp4mux fragment-duration=100 ! filesink location={path}")
            try:
                self.front_video_writer = cv2.VideoWriter(gst_pipe(front_path), cv2.CAP_GSTREAMER, 0, 20, (640,480), True)
                self.bottom_video_writer = cv2.VideoWriter(gst_pipe(bottom_path), cv2.CAP_GSTREAMER, 0, 20, (640,480), True)
                self.recording = True
                self.get_logger().info(f"🎥 录制启动: {front_path} / {bottom_path}")
            except Exception as e:
                self.get_logger().error(f"录制失败: {e}")
                self.front_video_writer = None
                self.bottom_video_writer = None
                self.recording = False

    def stop_video(self):
        with self.video_lock:
            if self.front_video_writer:
                self.front_video_writer.release()
                self.front_video_writer = None
            if self.bottom_video_writer:
                self.bottom_video_writer.release()
                self.bottom_video_writer = None
            self.recording = False

    def destroy_node(self):
        # 修复：等待舵机线程完成，避免资源释放冲突
        self._servo_shutdown = True
        if hasattr(self, '_servo_trigger_in_progress') and self._servo_trigger_in_progress:
            self.get_logger().info("等待舵机动作完成...")
            self.servo_event.wait(timeout=0.5)
        if self.servo_pwm is not None:
            self.servo_pwm.stop()
        GPIO.cleanup()
        self.stop_video()
        self.video_queue.put(None)
        for cap_attr in ('cap_front', 'cap_bottom'):
            cap = getattr(self, cap_attr, None)
            if cap is not None and cap.isOpened():
                cap.release()
        self.get_logger().info("资源已释放")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RobustSearchMission()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().warn("用户中断")
        node.trigger_emergency_land()
        for _ in range(3):
            rclpy.spin_once(node, timeout_sec=0.1)
        time.sleep(0.5)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()