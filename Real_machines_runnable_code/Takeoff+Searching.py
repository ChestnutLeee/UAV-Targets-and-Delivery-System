#!/usr/bin/env python3
"""
PX4 Offboard 稳健搜索节点（起飞安全版 + 45° 斜视修正 + 相机反装修正 + 视觉锁定反馈）
- 起飞逻辑完全对齐 Takeoff.py（实机验证）
- 视觉搜索：修正前视 45° 向下安装的偏航角计算
- 相机反装：自动旋转 180° 并录制正向视频
- 线程安全、ROS 时钟一致性、斜坡起飞
- 视觉锁定反馈：在 TARGET_HOLD 时输出目标偏移角度
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
from enum import Enum

# ---- CUDA 加速 ----
try:
    from numba import cuda, uint8
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False

if HAS_CUDA:
    @cuda.jit
    def preprocess_kernel(input_frame, output_gray):
        x, y = cuda.grid(2)
        if x < output_gray.shape[1] and y < output_gray.shape[0]:
            idx = (y * output_gray.shape[1] + x) * 3
            b = input_frame[idx]
            g = input_frame[idx + 1]
            r = input_frame[idx + 2]
            gray = 0.299 * r + 0.587 * g + 0.114 * b
            val = gray * 1.2 + 10.0
            output_gray[y, x] = uint8(max(0.0, min(255.0, val)))


class FlightState(Enum):
    WAIT_CONNECTION = 0
    WAIT_STABILIZE = 1
    ARMING = 2
    TAKEOFF = 3
    SEARCHING = 4
    TARGET_HOLD = 5
    EMERGENCY_LAND = 6


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
        self.declare_parameter('state_timeout', 15.0)
        self.declare_parameter('stabilize_wait', 3.0)
        self.declare_parameter('position_tolerance', 0.2)
        self.declare_parameter('camera_tilt_deg', 45.0)   # 相机下倾角度（度）

        self.target_alt = self.get_parameter('target_alt').value
        self.takeoff_speed = self.get_parameter('takeoff_speed').value
        self.search_yaw_rate_rad = math.radians(self.get_parameter('search_yaw_rate').value)
        self.target_tag_id = self.get_parameter('tag_id').value
        self.tag_detect_threshold = self.get_parameter('tag_detect_threshold').value
        self.heartbeat_timeout = self.get_parameter('heartbeat_timeout').value
        self.state_timeout = self.get_parameter('state_timeout').value
        self.stabilize_wait = self.get_parameter('stabilize_wait').value
        self.position_tolerance = self.get_parameter('position_tolerance').value
        self.camera_tilt_deg = self.get_parameter('camera_tilt_deg').value

        # 焦距：640 / (2*tan(77/2)) ≈ 402 px
        self.focal_length_px = 402.0

        # 预计算倾斜角三角函数
        self.cos_tilt = math.cos(-math.radians(self.camera_tilt_deg))
        self.sin_tilt = math.sin(-math.radians(self.camera_tilt_deg))

        # ---- 线程锁 ----
        self.state_lock = threading.Lock()
        self.data_lock = threading.Lock()
        self.video_lock = threading.Lock()
        self.vehicle_state_lock = threading.Lock()
        self.pos_lock = threading.Lock()

        # ---- 状态变量 ----
        self.flight_state = FlightState.WAIT_CONNECTION
        self.vehicle_state = State()
        self.current_pos = np.array([0.0, 0.0, 0.0])
        self.current_yaw = 0.0

        # 起飞原点（对齐 Takeoff.py：初始 None，稳定后赋值）
        self.takeoff_coords = [None, None]

        # 视觉检测（camera_loop 写入，control_loop 读取）
        self.tag_detected = False
        self.tag_detect_counter = 0
        self.raw_yaw_offset = 0.0
        self.filtered_yaw_offset = 0.0

        # 控制与安全
        self._state_enter_time = self.get_clock().now()
        self._last_heartbeat = time.time()
        self._takeoff_start_z = 0.0
        self._takeoff_start_time = None
        self._desired_takeoff_z = None
        self._search_target_yaw = 0.0
        self._last_search_time = None
        self._mode_req_sent = False
        self._arm_req_sent = False

        # 视觉丢失平滑保持
        self._last_valid_target_yaw = 0.0

        # 视频录制
        self.recording = False
        self.video_writer = None

        # ---- CUDA 预分配 ----
        if HAS_CUDA:
            self.d_frame = cuda.device_array((480 * 640 * 3,), dtype=np.uint8)
            self.d_gray = cuda.device_array((480, 640), dtype=np.uint8)

        # ---- ROS 接口 ----
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )
        self.state_sub = self.create_subscription(State, '/mavros/state', self.state_cb, qos)
        self.pos_sub = self.create_subscription(PoseStamped, '/mavros/local_position/pose', self.pos_cb, qos)
        self.local_pos_pub = self.create_publisher(PoseStamped, '/mavros/setpoint_position/local', 10)

        self.set_mode_cli = self.create_client(SetMode, '/mavros/set_mode')
        self.arm_cli = self.create_client(CommandBool, '/mavros/cmd/arming')

        # ---- 视觉初始化 ----
        self.init_camera()
        import apriltag
        options = apriltag.DetectorOptions(families="tag36h11")
        options.quad_decimate = 2.0
        self.detector = apriltag.Detector(options)

        # ---- 定时器与线程 ----
        self.control_timer = self.create_timer(0.02, self.control_loop)
        self.watchdog_timer = self.create_timer(0.5, self.watchdog_check)
        self.cam_thread = threading.Thread(target=self.camera_loop, daemon=True)
        self.cam_thread.start()

        self.get_logger().info("🚀 稳健搜索节点启动，等待飞控连接...")
        self.get_logger().warn("⚠️ 室内飞行前请确认 PX4 EKF2_HGT_MODE 已设置为 Range sensor")

    # ========== 线程安全访问器 ==========
    def get_mode(self):
        with self.vehicle_state_lock:
            return self.vehicle_state.mode

    def is_armed(self):
        with self.vehicle_state_lock:
            return self.vehicle_state.armed

    def is_connected(self):
        with self.vehicle_state_lock:
            return self.vehicle_state.connected

    def get_state(self):
        with self.state_lock:
            return self.flight_state

    def set_state(self, new_state):
        with self.state_lock:
            self.flight_state = new_state

    # ========== 回调 ==========
    def state_cb(self, msg):
        with self.vehicle_state_lock:
            self.vehicle_state = msg

    def pos_cb(self, msg):
        pos = msg.pose.position
        q = msg.pose.orientation
        yaw = math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))
        with self.pos_lock:
            self.current_pos = np.array([pos.x, pos.y, pos.z])
            self.current_yaw = yaw

    def init_camera(self):
        gst_str = ('nvarguscamerasrc ! video/x-raw(memory:NVMM), width=640, height=480, framerate=30/1 ! '
                   'nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink')
        self.cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            self.get_logger().warn("CSI 打开失败，尝试 USB")
            self.cap = cv2.VideoCapture(0)

    # ========== 摄像头线程（180° 旋转修正 + 45° 倾斜修正） ==========
    def camera_loop(self):
        while rclpy.ok():
            ret, frame = self.cap.read()
            if not ret:
                continue

            # ✅ 相机反装修正：旋转 180 度（上下左右均翻转，相当于顺时针 180°）
            frame = cv2.rotate(frame, cv2.ROTATE_180)

            # 录制视频（保存旋转后的正向画面）
            with self.video_lock:
                if self.recording and self.video_writer:
                    self.video_writer.write(cv2.resize(frame, (640, 480)))

            # 预处理（CUDA 加速或 CPU）
            if HAS_CUDA:
                self.d_frame.copy_to_device(frame.ravel())
                tpb = (16, 16)
                bpg = (math.ceil(640/16), math.ceil(480/16))
                preprocess_kernel[bpg, tpb](self.d_frame, self.d_gray)
                gray = self.d_gray.copy_to_host()
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            detections = self.detector.detect(gray)
            found = False
            y_off = 0.0
            for d in detections:
                if d.tag_id == self.target_tag_id:
                    # ---- 修正：45° 下倾相机的偏航角计算 ----
                    xn = (d.center[0] - 320.0) / self.focal_length_px
                    yn = (d.center[1] - 240.0) / self.focal_length_px

                    vx = xn
                    # vy = yn * self.cos_tilt - self.sin_tilt  # 垂直分量，此处不需要
                    vz = yn * self.sin_tilt + self.cos_tilt

                    y_off = math.atan2(vx, vz)
                    found = True
                    break

            # 更新检测计数器与滤波
            with self.data_lock:
                if found:
                    self.tag_detect_counter = min(10, self.tag_detect_counter + 1)
                else:
                    self.tag_detect_counter = max(0, self.tag_detect_counter - 1)
                self.tag_detected = (self.tag_detect_counter >= self.tag_detect_threshold)
                self.raw_yaw_offset = y_off
                self.filtered_yaw_offset = 0.85 * self.filtered_yaw_offset + 0.15 * y_off

    # ========== 主控制循环 ==========
    def control_loop(self):
        try:
            with self.pos_lock:
                curr_pos = self.current_pos.copy()
                curr_yaw = self.current_yaw

            if any(math.isnan(v) for v in curr_pos):
                return

            now = self.get_clock().now()
            current_state = self.get_state()

            # 紧急降落优先
            if current_state == FlightState.EMERGENCY_LAND:
                if self.get_mode() != 'AUTO.LAND':
                    self.request_mode('AUTO.LAND')
                self._last_heartbeat = time.time()
                return

            target_yaw = curr_yaw

            # ---- 状态机 ----
            if current_state == FlightState.WAIT_CONNECTION:
                if self.is_connected():
                    self.get_logger().info("✅ 飞控已连接")
                    self._enter_state(FlightState.WAIT_STABILIZE)

            elif current_state == FlightState.WAIT_STABILIZE:
                elapsed = (now - self._state_enter_time).nanoseconds / 1e9
                if elapsed > self.stabilize_wait:
                    self.takeoff_coords = [curr_pos[0], curr_pos[1]]
                    self.get_logger().info(f"📍 坐标锁定: {self.takeoff_coords}")
                    self._enter_state(FlightState.ARMING)

            elif current_state == FlightState.ARMING:
                if self.get_mode() != 'OFFBOARD':
                    self.request_mode('OFFBOARD')
                elif not self.is_armed():
                    self.request_arm(True)
                else:
                    self.get_logger().info("🎯 Offboard 已激活，进入斜坡起飞")
                    self.start_video()
                    self._enter_state(FlightState.TAKEOFF)

            elif current_state == FlightState.TAKEOFF:
                if self._takeoff_start_time is None:
                    self._takeoff_start_z = curr_pos[2]
                    self._takeoff_start_time = now
                    self.get_logger().info(f"开始缓慢起飞，起始高度: {self._takeoff_start_z:.2f} m")

                elapsed = (now - self._takeoff_start_time).nanoseconds / 1e9
                ramp_z = self._takeoff_start_z + self.takeoff_speed * elapsed
                self._desired_takeoff_z = min(ramp_z, self.target_alt)

                if abs(curr_pos[2] - self.target_alt) < self.position_tolerance:
                    self.get_logger().info("✅ 到达目标高度，进入搜索")
                    self._enter_state(FlightState.SEARCHING)
                    self._search_target_yaw = curr_yaw
                    self._last_search_time = now

            elif current_state == FlightState.SEARCHING:
                with self.data_lock:
                    detected = self.tag_detected
                    filt_off = self.filtered_yaw_offset
                if detected:
                    self._last_valid_target_yaw = curr_yaw + filt_off
                    self._enter_state(FlightState.TARGET_HOLD)
                else:
                    if self._last_search_time is None:
                        self._last_search_time = now
                    dt = (now - self._last_search_time).nanoseconds / 1e9
                    self._search_target_yaw += self.search_yaw_rate_rad * dt
                    while self._search_target_yaw > math.pi:
                        self._search_target_yaw -= 2.0 * math.pi
                    while self._search_target_yaw < -math.pi:
                        self._search_target_yaw += 2.0 * math.pi
                    self._last_search_time = now
                    target_yaw = self._search_target_yaw

            elif current_state == FlightState.TARGET_HOLD:
                with self.data_lock:
                    detected = self.tag_detected
                    filt_off = self.filtered_yaw_offset
                if detected:
                    target_yaw = curr_yaw + filt_off
                    self._last_valid_target_yaw = target_yaw
                    # 视觉锁定反馈
                    self.get_logger().info(
                        f"🎯 目标锁定！偏移: {math.degrees(filt_off):.2f}°",
                        throttle_duration_sec=1.0
                    )
                else:
                    if (now - self._state_enter_time).nanoseconds / 1e9 > 5.0:
                        self._enter_state(FlightState.SEARCHING)
                    else:
                        target_yaw = self._last_valid_target_yaw

            # 统一发布
            self.publish_setpoint(curr_pos[0], curr_pos[1], curr_pos[2], target_yaw)
            self._last_heartbeat = time.time()

        except Exception as e:
            self.get_logger().error(f"❌ 控制循环异常: {e}", throttle_duration_sec=1.0)

    # ========== 安全监控 ==========
    def watchdog_check(self):
        if time.time() - self._last_heartbeat > self.heartbeat_timeout:
            self.get_logger().error("⏰ 控制循环丢失，触发紧急降落")
            self.trigger_emergency_land()
            return

        now = self.get_clock().now()
        elapsed = (now - self._state_enter_time).nanoseconds / 1e9
        current_state = self.get_state()

        if current_state in (FlightState.TARGET_HOLD, FlightState.EMERGENCY_LAND):
            return

        timeout = self.state_timeout
        if current_state == FlightState.SEARCHING:
            timeout = self.state_timeout * 2

        if elapsed > timeout:
            self.get_logger().error(f"⏰ 状态 '{current_state.name}' 超时，触发降落")
            self.trigger_emergency_land()

    def trigger_emergency_land(self):
        if self.get_state() == FlightState.EMERGENCY_LAND:
            return
        self.get_logger().error("🚨 切换至紧急降落")
        self._enter_state(FlightState.EMERGENCY_LAND)
        self.request_mode('AUTO.LAND')
        self.stop_video()

    def _enter_state(self, new_state):
        if self.get_state() == new_state:
            return
        self.get_logger().info(f"状态切换: {self.get_state().name} → {new_state.name}")
        self.set_state(new_state)
        self._state_enter_time = self.get_clock().now()
        self._mode_req_sent = False
        self._arm_req_sent = False

        if new_state != FlightState.TAKEOFF:
            self._desired_takeoff_z = None
            self._takeoff_start_time = None
        if new_state == FlightState.SEARCHING:
            with self.pos_lock:
                self._search_target_yaw = self.current_yaw
            self._last_search_time = self.get_clock().now()

    # ========== 模式 / 解锁服务 ==========
    def request_mode(self, target_mode):
        if self._mode_req_sent:
            return
        self.get_logger().info(f"→ 请求模式: {target_mode}")
        req = SetMode.Request()
        req.custom_mode = target_mode
        future = self.set_mode_cli.call_async(req)
        future.add_done_callback(lambda f, m=target_mode: self._mode_response_callback(f, m))
        self._mode_req_sent = True

    def _mode_response_callback(self, future, target_mode):
        try:
            result = future.result()
            if result is not None and result.mode_sent:
                self.get_logger().info(f"✅ {target_mode} 模式请求已接受")
            else:
                current_mode = self.get_mode()
                if current_mode == target_mode:
                    self.get_logger().info(f"⚠️ {target_mode} 模式实际已进入")
                else:
                    self.get_logger().error(f"❌ 模式切换失败 (当前: {current_mode})，允许重试")
                    self._mode_req_sent = False
        except Exception as e:
            self.get_logger().error(f"❌ 模式服务异常: {e}，允许重试")
            self._mode_req_sent = False

    def request_arm(self, value):
        if self._arm_req_sent:
            return
        self.get_logger().info(f"→ 请求解锁: {value}")
        req = CommandBool.Request()
        req.value = value
        future = self.arm_cli.call_async(req)
        future.add_done_callback(lambda f, v=value: self._arm_response_callback(f, v))
        self._arm_req_sent = True

    def _arm_response_callback(self, future, value):
        try:
            result = future.result()
            if result is not None and result.success:
                self.get_logger().info("✅ 解锁服务成功")
            else:
                if self.is_armed() == bool(value):
                    self.get_logger().info("⚠️ 解锁服务未确认，但状态已匹配")
                else:
                    self.get_logger().error(f"❌ 解锁失败，当前 armed={self.is_armed()}，允许重试")
                    self._arm_req_sent = False
        except Exception as e:
            self.get_logger().error(f"❌ 解锁服务异常: {e}，允许重试")
            self._arm_req_sent = False

    # ========== 发布（包含高度/偏航控制） ==========
    def publish_setpoint(self, curr_x, curr_y, curr_z, yaw):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"

        # 水平位置锁定（对齐 Takeoff.py）
        if self.takeoff_coords[0] is not None:
            msg.pose.position.x = self.takeoff_coords[0]
            msg.pose.position.y = self.takeoff_coords[1]
        else:
            msg.pose.position.x = curr_x
            msg.pose.position.y = curr_y

        # 高度控制
        current_state = self.get_state()
        if current_state == FlightState.TAKEOFF and self._desired_takeoff_z is not None:
            msg.pose.position.z = self._desired_takeoff_z
        elif current_state in (FlightState.SEARCHING, FlightState.TARGET_HOLD):
            msg.pose.position.z = self.target_alt
        else:
            msg.pose.position.z = curr_z

        # 偏航：最短路径
        with self.pos_lock:
            curr_yaw = self.current_yaw
        diff = yaw - curr_yaw
        while diff > math.pi:
            diff -= 2.0 * math.pi
        while diff < -math.pi:
            diff += 2.0 * math.pi
        final_yaw = curr_yaw + diff
        msg.pose.orientation.z = math.sin(final_yaw / 2.0)
        msg.pose.orientation.w = math.cos(final_yaw / 2.0)

        self.local_pos_pub.publish(msg)

    # ========== 视频录制（保存到 /home/jetson/桌面） ==========
    def start_video(self):
        with self.video_lock:
            if not self.recording:
                save_dir = "/home/jetson/桌面"
                os.makedirs(save_dir, exist_ok=True)
                path = os.path.join(save_dir, f"flight_{int(time.time())}.mp4")
                self.video_writer = cv2.VideoWriter(
                    path, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (640, 480))
                if self.video_writer.isOpened():
                    self.recording = True
                    self.get_logger().info(f"🎥 录制开始: {path}")
                else:
                    self.get_logger().error("无法创建视频文件")

    def stop_video(self):
        with self.video_lock:
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
                self.recording = False

    def destroy_node(self):
        self.stop_video()
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
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
        node.stop_video()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()