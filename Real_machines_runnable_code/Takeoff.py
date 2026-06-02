#!/usr/bin/env python3
"""
PX4 Offboard 鲁棒起飞节点（最终安全版 - 斜坡起飞 + 多模式回调修复 + 线程安全）
适用：ROS 2 Humble + MAVROS + PX4（1.13+ / 室内光流 H-flow）
核心特性：
- 斜坡起飞：TAKEOFF 阶段目标高度线性增加，避免瞬间阶跃，配合 MPC_Z_VEL_MAX_UP 双重保障
- 模式请求回调绑定目标模式，紧急降落重试永不阻塞
- 线程安全保护关键状态
- 全轴 NaN 检查
- ROS 2 参数动态配置
"""

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode

import time
import threading
import math
from enum import Enum


class FlightState(Enum):
    WAIT_CONNECTION = 0
    WAIT_STABILIZE = 1
    ARMING = 2
    TAKEOFF = 3
    HOVER = 4
    EMERGENCY_LAND = 5


class RobustDroneTakeoff(Node):
    def __init__(self):
        super().__init__('robust_drone_takeoff')

        # ------- 参数声明 -------
        self.declare_parameter('target_alt', 1.5)
        self.declare_parameter('position_tolerance', 0.2)
        self.declare_parameter('stabilize_wait', 3.0)
        self.declare_parameter('state_timeout', 15.0)
        self.declare_parameter('heartbeat_timeout', 1.0)
        self.declare_parameter('takeoff_speed', 0.5)          # 爬升速率 m/s

        # ------- 内部状态（线程安全） -------
        self.flight_state = FlightState.WAIT_CONNECTION
        self.state_lock = threading.Lock()
        self.vehicle_state = State()
        self.vehicle_state_lock = threading.Lock()
        self.current_pos = [0.0, 0.0, 0.0]
        self.pos_lock = threading.Lock()

        self.takeoff_coords = [None, None]
        self._mode_req_sent = False
        self._arm_req_sent = False
        self._state_enter_time = self.get_clock().now()
        self._last_heartbeat = time.time()

        # 斜坡起飞相关变量
        self._takeoff_start_z = 0.0
        self._takeoff_start_time = None
        self._desired_takeoff_z = None

        # 读取参数
        self.target_alt = self.get_parameter('target_alt').value
        self.position_tolerance = self.get_parameter('position_tolerance').value
        self.stabilize_wait = self.get_parameter('stabilize_wait').value
        self.state_timeout = self.get_parameter('state_timeout').value
        self.heartbeat_timeout = self.get_parameter('heartbeat_timeout').value
        self.takeoff_speed = self.get_parameter('takeoff_speed').value

        # QoS 配置
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # ------- 订阅与发布 -------
        self.state_sub = self.create_subscription(State, '/mavros/state', self.state_cb, qos)
        self.pos_sub = self.create_subscription(PoseStamped, '/mavros/local_position/pose', self.pos_cb, qos)
        self.local_pos_pub = self.create_publisher(PoseStamped, '/mavros/setpoint_position/local', 10)

        # ------- 服务客户端 -------
        self.set_mode_cli = self.create_client(SetMode, '/mavros/set_mode')
        self.arm_cli = self.create_client(CommandBool, '/mavros/cmd/arming')

        # ------- 定时器 -------
        self.timer = self.create_timer(0.02, self.control_loop)
        self.watchdog_timer = self.create_timer(0.5, self.watchdog_check)

        self.get_logger().info("🚀 最终安全版起飞节点启动，等待飞控连接...")

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

    # ========== 回调函数 ==========
    def state_cb(self, msg):
        with self.vehicle_state_lock:
            self.vehicle_state = msg

    def pos_cb(self, msg):
        with self.pos_lock:
            self.current_pos = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]

    # ========== 核心逻辑 ==========
    def watchdog_check(self):
        if time.time() - self._last_heartbeat > self.heartbeat_timeout:
            self.get_logger().error("⏰ 控制循环丢失，触发紧急降落")
            self.trigger_emergency_land()
            return

        now = self.get_clock().now()
        elapsed = (now - self._state_enter_time).nanoseconds / 1e9
        current_state = self.get_state()
        if current_state not in (FlightState.HOVER, FlightState.EMERGENCY_LAND) and elapsed > self.state_timeout:
            self.get_logger().error(f"⏰ 状态 '{current_state.name}' 超时，触发降落")
            self.trigger_emergency_land()

    def _enter_state(self, new_state):
        if self.get_state() == new_state:
            return
        self.get_logger().info(f"状态切换: {self.get_state().name} → {new_state.name}")
        self.set_state(new_state)
        self._state_enter_time = self.get_clock().now()
        self._mode_req_sent = False
        self._arm_req_sent = False

        # 状态切换时清理斜坡变量
        if new_state != FlightState.TAKEOFF:
            self._desired_takeoff_z = None
            self._takeoff_start_time = None

    def trigger_emergency_land(self):
        if self.get_state() == FlightState.EMERGENCY_LAND:
            return
        self.get_logger().error("🚨 切换至紧急降落流程 (AUTO.LAND)，立即发送请求")
        self._enter_state(FlightState.EMERGENCY_LAND)
        self.request_mode('AUTO.LAND')

    def request_mode(self, target_mode):
        if self._mode_req_sent:
            return
        self.get_logger().info(f"→ 请求模式: {target_mode}")
        req = SetMode.Request()
        req.custom_mode = target_mode
        future = self.set_mode_cli.call_async(req)
        # 闭包绑定目标模式
        future.add_done_callback(lambda f, mode=target_mode: self._mode_response_callback(f, mode))
        self._mode_req_sent = True

    def _mode_response_callback(self, future, target_mode):
        try:
            result = future.result()
            if result is not None and result.mode_sent:
                self.get_logger().info(f"✅ {target_mode} 模式请求已接受")
            else:
                current_mode = self.get_mode()
                if current_mode == target_mode:
                    self.get_logger().info(f"⚠️ {target_mode} 服务未返回成功，但飞控实际已进入")
                else:
                    self.get_logger().error(
                        f"❌ 模式切换失败 (当前: {current_mode}, 期望: {target_mode})，允许重试")
                    self._mode_req_sent = False
        except Exception as e:
            self.get_logger().error(f"❌ 模式服务调用异常: {e}，允许重试")
            self._mode_req_sent = False

    def request_arm(self, value):
        if self._arm_req_sent:
            return
        self.get_logger().info(f"→ 请求解锁: {value}")
        req = CommandBool.Request()
        req.value = value
        future = self.arm_cli.call_async(req)
        future.add_done_callback(lambda f, val=value: self._arm_response_callback(f, val))
        self._arm_req_sent = True

    def _arm_response_callback(self, future, value):
        try:
            result = future.result()
            if result is not None and result.success:
                self.get_logger().info("✅ 解锁服务成功")
            else:
                if self.is_armed() == bool(value):
                    self.get_logger().info("⚠️ 解锁服务未确认，但飞控状态已匹配")
                else:
                    self.get_logger().error(f"❌ 解锁服务失败，当前 armed={self.is_armed()}，允许重试")
                    self._arm_req_sent = False
        except Exception as e:
            self.get_logger().error(f"❌ 解锁服务异常: {e}，允许重试")
            self._arm_req_sent = False

    def publish_setpoint(self, curr_x, curr_y, curr_z):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"

        # 水平坐标锁定
        if self.takeoff_coords[0] is not None:
            msg.pose.position.x = self.takeoff_coords[0]
            msg.pose.position.y = self.takeoff_coords[1]
        else:
            msg.pose.position.x = curr_x
            msg.pose.position.y = curr_y

        # 高度控制（斜坡起飞优先）
        current_state = self.get_state()
        if current_state == FlightState.TAKEOFF and self._desired_takeoff_z is not None:
            msg.pose.position.z = self._desired_takeoff_z
        elif current_state == FlightState.HOVER:
            msg.pose.position.z = self.target_alt
        else:
            msg.pose.position.z = curr_z

        msg.pose.orientation.w = 1.0
        self.local_pos_pub.publish(msg)

    def control_loop(self):
        try:
            with self.pos_lock:
                curr_pos = list(self.current_pos)

            # 全轴 NaN 检查
            if any(math.isnan(v) for v in curr_pos):
                return

            self.publish_setpoint(curr_pos[0], curr_pos[1], curr_pos[2])
            self._last_heartbeat = time.time()
            now = self.get_clock().now()
            current_state = self.get_state()

            # 紧急降落劫持
            if current_state == FlightState.EMERGENCY_LAND:
                if self.get_mode() != 'AUTO.LAND':
                    self.request_mode('AUTO.LAND')
                return

            # 正常状态机
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
                    self._enter_state(FlightState.TAKEOFF)

            elif current_state == FlightState.TAKEOFF:
                # 斜坡起飞逻辑
                if self._takeoff_start_time is None:
                    self._takeoff_start_z = curr_pos[2]
                    self._takeoff_start_time = now
                    self.get_logger().info(f"开始缓慢起飞，起始高度: {self._takeoff_start_z:.2f} m")

                elapsed = (now - self._takeoff_start_time).nanoseconds / 1e9
                ramp_z = self._takeoff_start_z + self.takeoff_speed * elapsed
                self._desired_takeoff_z = min(ramp_z, self.target_alt)

                if abs(curr_pos[2] - self.target_alt) < self.position_tolerance:
                    self.get_logger().info("✅ 到达目标高度，悬停")
                    self._enter_state(FlightState.HOVER)

            elif current_state == FlightState.HOVER:
                pass

        except Exception as e:
            self.get_logger().error(f"❌ 控制循环异常: {e}", throttle_duration_sec=1.0)


def main(args=None):
    rclpy.init(args=args)
    node = RobustDroneTakeoff()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().warn("⏹️ 用户中断，执行降落")
        req = SetMode.Request()
        req.custom_mode = 'AUTO.LAND'
        node.set_mode_cli.call_async(req)
        time.sleep(0.5)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()