#!/usr/bin/env python3
"""
PX4 Offboard 搜索 + 逼近 + 精准悬停 + 舵机投放 + 自动返航降落 v1.4.9
修复：
- ARMING 状态中 OFFBOARD 模式确认超时强制进入，避免卡死
- 增加模式切换重试与超时保护
- 默认 state_timeout 提升至 30 秒，减少误触发
"""

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from geometry_msgs.msg import PoseStamped
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
    HOVER_ABOVE_TAG = 6
    LANDING = 7
    EMERGENCY_LAND = 8
    RETURN_TO_LAND = 9


class RobustSearchMission(Node):
    def __init__(self):
        super().__init__('robust_search_mission')

        # ---- 参数配置 ----
        self.declare_parameter('target_alt', 1.5)
        self.declare_parameter('hover_alt', 1.0)
        self.declare_parameter('takeoff_speed', 0.5)
        self.declare_parameter('search_yaw_rate', 20.0)
        self.declare_parameter('tag_id', 41)
        self.declare_parameter('tag_detect_threshold', 3)
        self.declare_parameter('heartbeat_timeout', 1.0)
        self.declare_parameter('state_timeout', 30.0)          # 增加超时
        self.declare_parameter('max_takeoff_time', 60.0)
        self.declare_parameter('stabilize_wait', 3.0)
        self.declare_parameter('position_tolerance', 0.2)
        self.declare_parameter('camera_tilt_deg', 45.0)
        self.declare_parameter('approach_speed', 0.5)
        self.declare_parameter('align_duration', 2.0)
        self.declare_parameter('lost_search_duration', 1.0)
        self.declare_parameter('approach_descent_duration', 5.0)
        self.declare_parameter('yaw_rate_limit', 90.0)
        self.declare_parameter('contrast_gain', 1.2)
        self.declare_parameter('brightness_offset', 10.0)
        self.declare_parameter('lost_approach_wait_duration', 2.0)
        self.declare_parameter('landing_hold_duration', 3.0)
        self.declare_parameter('landing_position_tolerance', 0.1)
        self.declare_parameter('use_cuda', False)
        self.declare_parameter('bottom_cam_dx_sign', 1)
        self.declare_parameter('bottom_cam_dy_sign', -1)
        self.declare_parameter('swap_bottom_axes', False)
        self.declare_parameter('approach_xy_smoothing', 0.8)
        self.declare_parameter('altitude_jump_threshold', 0.5)
        self.declare_parameter('low_alt_threshold', 0.5)
        self.declare_parameter('alt_jump_count_threshold', 5)
        self.declare_parameter('use_hw_encoding', False)
        self.declare_parameter('vision_timeout', 3.0)
        self.declare_parameter('vision_staleness_limit', 0.2)
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
        self.declare_parameter('hover_filter_alpha', 0.7)
        self.declare_parameter('rtl_altitude', 1.5)
        self.declare_parameter('rtl_timeout', 30.0)

        # 读取所有参数
        self.target_alt = self.get_parameter('target_alt').value
        self.hover_alt = self.get_parameter('hover_alt').value
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
        self.approach_speed = self.get_parameter('approach_speed').value
        self.align_duration = self.get_parameter('align_duration').value
        self.lost_search_duration = self.get_parameter('lost_search_duration').value
        self.approach_descent_duration = self.get_parameter('approach_descent_duration').value
        self.yaw_rate_limit = math.radians(self.get_parameter('yaw_rate_limit').value)
        self.contrast_gain = self.get_parameter('contrast_gain').value
        self.brightness_offset = self.get_parameter('brightness_offset').value
        self.lost_approach_wait_duration = self.get_parameter('lost_approach_wait_duration').value
        self.landing_hold_duration = self.get_parameter('landing_hold_duration').value
        self.landing_position_tolerance = self.get_parameter('landing_position_tolerance').value
        self.use_cuda = self.get_parameter('use_cuda').value and HAS_CUDA
        self.bottom_dx_sign = self.get_parameter('bottom_cam_dx_sign').value
        self.bottom_dy_sign = self.get_parameter('bottom_cam_dy_sign').value
        self.swap_bottom_axes = self.get_parameter('swap_bottom_axes').value
        self.approach_xy_smoothing = self.get_parameter('approach_xy_smoothing').value
        self.altitude_jump_threshold = self.get_parameter('altitude_jump_threshold').value
        self.low_alt_threshold = self.get_parameter('low_alt_threshold').value
        self.alt_jump_count_threshold = self.get_parameter('alt_jump_count_threshold').value
        self.use_hw_encoding = self.get_parameter('use_hw_encoding').value
        self.vision_timeout = self.get_parameter('vision_timeout').value
        self.vision_staleness_limit = self.get_parameter('vision_staleness_limit').value
        self.min_disk_space_mb = self.get_parameter('min_disk_space_mb').value
        self.hover_filter_alpha = self.get_parameter('hover_filter_alpha').value
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

        # 警告
        if abs(self.front_focal_length - 703.0) < 1e-2 and abs(self.bottom_focal_length - 722.0) < 1e-2:
            self.get_logger().warn("⚠️ 焦距仍为默认值，请根据实际标定修改！")
        if self.approach_speed > 0.8:
            self.get_logger().warn(f"逼近速度 {self.approach_speed} m/s 可能过快")
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
        self.detector_lock = threading.Lock()
        self.servo_lock = threading.Lock()

        # 状态变量
        self.flight_state = FlightState.WAIT_CONNECTION
        self.vehicle_state = State()
        self.current_pos = np.array([0.0, 0.0, 0.0])
        self.current_yaw = 0.0
        self.takeoff_coords = [None, None]

        self.front_tag_detected = False
        self.front_tag_detect_counter = 0
        self.front_yaw_offset = 0.0

        self.bottom_tag_detected = False
        self.bottom_tag_detect_counter = 0
        self.bottom_yaw_offset = 0.0
        self.bottom_pixel_offset = (0.0, 0.0)
        self._bottom_pose_time = 0.0

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

        self._approach_desired_xy = None
        self._approach_smooth_xy = None
        self._approach_align_start = None
        self._approach_forward_start = None
        self._approach_lost_time = None
        self._last_known_direction = None
        self._approach_substate = 0
        self._hover_xy = None
        self._landing_ready_time = None

        self.servo_triggered = False
        self.servo_action_complete = False
        self.servo_event = threading.Event()
        self.servo_pwm = None
        self._servo_trigger_in_progress = False

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

        self.init_servo()

        # ROS 接口
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE,
                         history=HistoryPolicy.KEEP_LAST, depth=5)
        self.state_sub = self.create_subscription(State, '/mavros/state', self.state_cb, qos)
        self.pos_sub = self.create_subscription(PoseStamped, '/mavros/local_position/pose', self.pos_cb, qos)
        self.local_pos_pub = self.create_publisher(PoseStamped, '/mavros/setpoint_position/local', 10)
        self.set_mode_cli = self.create_client(SetMode, '/mavros/set_mode')
        self.arm_cli = self.create_client(CommandBool, '/mavros/cmd/arming')

        self.init_cameras()
        import apriltag
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

        self.get_logger().info("🚀 v1.4.9：修复 ARMING 卡死，模式超时强制进入")
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
                time.sleep(self.servo_hold_time)
                self.get_logger().info("🎁 投放完成：关闭舵机")
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
            now = time.time()
            if now - self._last_video_log_time > 1.0:
                self.get_logger().info(f"🎞️ 视频队列: {self.video_queue.qsize()}/100, 丢弃: {self._video_dropped}")
                self._last_video_log_time = now
            if self.recording and now - self._last_disk_check_time > 10.0:
                self._check_disk_space()
                self._last_disk_check_time = now

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
            self._front_last_frame_time = time.time()
            detections, yaw_off, _ = self.detect_tag(frame, True, d_frame, d_gray, adaptive_gain)
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
            self._bottom_last_frame_time = time.time()
            detections, yaw_off, pix_off = self.detect_tag(frame, False, d_frame, d_gray, adaptive_gain)
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
                self.bottom_yaw_offset = yaw_off if yaw_off is not None else 0.0
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
        focal = self.front_focal_length if is_front else self.bottom_focal_length
        for d in detections:
            if d.tag_id == self.target_tag_id:
                dx = d.center[0] - 320.0
                dy = d.center[1] - 240.0
                if is_front:
                    xn = dx / focal
                    yn = dy / focal
                    yaw_off = math.atan2(-xn, yn*self.sin_tilt + self.cos_tilt)
                else:
                    yaw_off = -math.atan2(dx, -dy)
                    pixel_off = (dx, dy)
                break
        return detections, yaw_off, pixel_off

    # ========== 主控制循环 ==========
    def control_loop(self):
        try:
            with self.pos_lock:
                curr_pos = self.current_pos.copy()
                curr_yaw = self.current_yaw
                alt = curr_pos[2]
            if any(math.isnan(v) for v in curr_pos):
                return

            # 高度跳变检测（仅下降）
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
                # 使用临时变量防止重复请求
                if not hasattr(self, '_arming_step'):
                    self._arming_step = 0          # 0: 请求OFFBOARD, 1: 请求解锁, 2: 完成
                    self._arming_enter_time = now

                if self._arming_step == 0:
                    if self.get_mode() != 'OFFBOARD':
                        self.request_mode('OFFBOARD')
                    # 等待模式切换或超时2秒
                    elapsed_mode = (now - self._arming_enter_time).nanoseconds / 1e9
                    if self.get_mode() == 'OFFBOARD' or elapsed_mode > 2.0:
                        self._arming_step = 1
                        self.get_logger().info(f"OFFBOARD 模式确认 (mode={self.get_mode()}), 进入解锁步骤")
                elif self._arming_step == 1:
                    if not self.is_armed():
                        self.request_arm(True)
                    # 等待解锁或超时3秒
                    elapsed_arm = (now - self._arming_enter_time).nanoseconds / 1e9
                    if self.is_armed() or elapsed_arm > 3.0:
                        if not self.is_armed():
                            self.get_logger().warn("解锁未确认，强制继续")
                        self._arming_step = 2
                elif self._arming_step == 2:
                    # 进入起飞
                    self.start_video()
                    self._enter_state(FlightState.TAKEOFF)
                    delattr(self, '_arming_step')
                    delattr(self, '_arming_enter_time')

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
                    bottom_det = self.bottom_tag_detected
                if bottom_det:
                    self._enter_state(FlightState.HOVER_ABOVE_TAG)
                elif front_det:
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

            elif current_state == FlightState.APPROACHING:
                with self.data_lock:
                    front_det = self.front_tag_detected
                    bottom_det = self.bottom_tag_detected
                    front_off = self.front_yaw_offset
                if bottom_det:
                    self._enter_state(FlightState.HOVER_ABOVE_TAG)
                    self.publish_setpoint(curr_pos, desired_xy, target_yaw, z_desired)
                    self._last_heartbeat = time.time()
                    return

                if self._approach_substate == 0:
                    if self._approach_desired_xy is not None:
                        desired_xy = self._approach_desired_xy.copy()
                    target_yaw = curr_yaw + front_off
                    if abs(front_off) < 0.1:
                        if self._approach_align_start is None:
                            self._approach_align_start = now
                        elif (now - self._approach_align_start).nanoseconds / 1e9 > self.align_duration:
                            self._approach_substate = 1
                            self._approach_forward_start = now
                    else:
                        self._approach_align_start = None
                elif self._approach_substate == 1:
                    forward_dir = np.array([math.cos(curr_yaw), math.sin(curr_yaw)])
                    if self._approach_smooth_xy is None:
                        self._approach_smooth_xy = curr_pos[:2].copy()
                    if self._approach_forward_start is not None:
                        elapsed_f = (now - self._approach_forward_start).nanoseconds / 1e9
                        ratio = min(1.0, elapsed_f / self.approach_descent_duration)
                        z_desired = self.target_alt - (self.target_alt - self.hover_alt) * ratio
                    else:
                        z_desired = self.target_alt
                    if front_det:
                        target_yaw = curr_yaw + front_off
                        self._last_known_direction = forward_dir
                        self._approach_lost_time = None
                        self._approach_smooth_xy += forward_dir * (self.approach_speed * dt)
                        self._approach_desired_xy = self._approach_smooth_xy.copy()
                    else:
                        current_xy = curr_pos[:2]
                        if self._approach_lost_time is None:
                            self._approach_lost_time = now
                        elif (now - self._approach_lost_time).nanoseconds / 1e9 > self.lost_approach_wait_duration:
                            self.takeoff_coords = [curr_pos[0], curr_pos[1]]
                            self._enter_state(FlightState.SEARCHING)
                            self.publish_setpoint(curr_pos, desired_xy, target_yaw, z_desired)
                            self._last_heartbeat = time.time()
                            return
                        to_current = current_xy - self._approach_desired_xy
                        dist = np.linalg.norm(to_current)
                        if dist > 0.01:
                            move = to_current / dist * min(0.2 * dt, dist)
                            self._approach_desired_xy += move
                            self._approach_smooth_xy = self._approach_desired_xy.copy()
                        else:
                            self._approach_desired_xy = current_xy.copy()
                            self._approach_smooth_xy = current_xy.copy()
                        if self._last_known_direction is not None:
                            target_yaw = math.atan2(self._last_known_direction[1], self._last_known_direction[0])
                        else:
                            target_yaw = curr_yaw
                    desired_xy = self._approach_desired_xy

            elif current_state == FlightState.HOVER_ABOVE_TAG:
                with self.data_lock:
                    bottom_det = self.bottom_tag_detected
                    bottom_yaw_off = self.bottom_yaw_offset
                    dx, dy = self.bottom_pixel_offset
                    pose_time = self._bottom_pose_time
                data_stale = (time.time() - pose_time) > self.vision_staleness_limit
                if bottom_det and data_stale:
                    self.get_logger().warn(f"下视数据过时，视为丢失", throttle_duration_sec=1.0)
                    bottom_det = False

                # 投放完成 -> 返航
                if self.servo_triggered and self.servo_action_complete:
                    self.get_logger().info("✅ 投放完成，启动 AUTO.RTL")
                    self._enter_state(FlightState.RETURN_TO_LAND)
                    self.publish_setpoint(curr_pos, desired_xy, target_yaw, z_desired)
                    self._last_heartbeat = time.time()
                    return

                if not bottom_det:
                    self.get_logger().warn("下视丢失目标，返回搜索")
                    self.takeoff_coords = [curr_pos[0], curr_pos[1]]
                    self._enter_state(FlightState.SEARCHING)
                    self.publish_setpoint(curr_pos, desired_xy, target_yaw, z_desired)
                    self._last_heartbeat = time.time()
                    return
                else:
                    target_yaw = curr_yaw + bottom_yaw_off
                    effective_alt = max(alt, 0.1)
                    if self.swap_bottom_axes:
                        dx, dy = dy, dx
                    body_x = self.bottom_dy_sign * dy / self.bottom_focal_length * effective_alt
                    body_y = self.bottom_dx_sign * dx / self.bottom_focal_length * effective_alt
                    cos_y = math.cos(curr_yaw); sin_y = math.sin(curr_yaw)
                    dx_enu = body_x * cos_y - body_y * sin_y
                    dy_enu = body_x * sin_y + body_y * cos_y
                    raw_hover_xy = np.array([curr_pos[0] + dx_enu, curr_pos[1] + dy_enu])
                    if self._hover_xy is None:
                        self._hover_xy = raw_hover_xy.copy()
                    else:
                        self._hover_xy = self.hover_filter_alpha * self._hover_xy + (1 - self.hover_filter_alpha) * raw_hover_xy
                    desired_xy = self._hover_xy.copy()
                    z_desired = self.hover_alt
                    if not self.servo_triggered and not self._servo_trigger_in_progress:
                        pos_error = math.hypot(dx_enu, dy_enu)
                        yaw_error = abs(bottom_yaw_off)
                        if pos_error < 0.1 and yaw_error < 0.05:
                            self.trigger_servo_drop()

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
        vision_now = time.time()
        if self._front_last_frame_time > 0 and vision_now - self._front_last_frame_time > self.vision_timeout:
            self.get_logger().error("前视摄像头无新帧，紧急降落")
            self.trigger_emergency_land()
            return
        if self._bottom_last_frame_time > 0 and vision_now - self._bottom_last_frame_time > self.vision_timeout:
            self.get_logger().error("下视摄像头无新帧，紧急降落")
            self.trigger_emergency_land()
            return
        elapsed = (now - self._state_enter_time).nanoseconds / 1e9
        state = self.get_state()
        if state in (FlightState.HOVER_ABOVE_TAG, FlightState.EMERGENCY_LAND, FlightState.LANDING, FlightState.RETURN_TO_LAND):
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
            elif new_state == FlightState.APPROACHING:
                self._approach_substate = 0
                self._approach_align_start = None
                self._approach_forward_start = None
                self._approach_lost_time = None
                self._last_known_direction = None
                self._approach_smooth_xy = None
                with self.pos_lock:
                    self._approach_desired_xy = np.array([self.current_pos[0], self.current_pos[1]])
            elif new_state == FlightState.HOVER_ABOVE_TAG:
                self._landing_ready_time = None
                with self.pos_lock:
                    self._hover_xy = np.array([self.current_pos[0], self.current_pos[1]])
            elif new_state == FlightState.RETURN_TO_LAND:
                self._rtl_start_time = None
            elif new_state == FlightState.LANDING:
                self.request_mode('AUTO.LAND')
                self.stop_video()

    # ========== 模式/解锁（带重试） ==========
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
            elif st in (FlightState.SEARCHING, FlightState.APPROACHING, FlightState.HOVER_ABOVE_TAG):
                msg.pose.position.z = self.target_alt
            else:
                msg.pose.position.z = float(curr_pos[2])
        msg.pose.orientation.z = math.sin(yaw/2)
        msg.pose.orientation.w = math.cos(yaw/2)
        self.local_pos_pub.publish(msg)

    # ========== 视频录制 ==========
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
                    return (f"appsrc ! videoconvert ! video/x-raw,format=I420 ! "
                            f"nvvidconv ! nvh264enc bitrate=3000 preset=low-latency ! "
                            f"h264parse ! mp4mux fragment-duration=100 ! filesink location={path}")
                else:
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
    finally:
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()