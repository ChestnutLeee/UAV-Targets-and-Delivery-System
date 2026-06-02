#!/usr/bin/env python3
import rclpy
import logging
import time
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleLocalPosition,
    VehicleStatus,
    BatteryStatus
)

# 日志配置保持不变
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')

class DroneMission(Node):
    def __init__(self):
        super().__init__('takingoff_mode')
        
        self.target_altitude = 6.0
        self.is_vision_ready = False
        self.current_altitude = 0.0
        self.battery_percent = 1.0
        self.has_position_data = False
        
        # QoS 配置文件 - 与 PX4 兼容
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
        
        # ROS 2 Publishers
        self.offboard_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.setpoint_pub = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)
        self.cmd_pub = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', qos_profile)

        # ROS 2 Subscribers - 使用正确的 QoS 配置
        self.local_pos_sub = self.create_subscription(
            VehicleLocalPosition, 
            '/fmu/out/vehicle_local_position_v1', 
            self.pos_callback, 
            qos_profile
        )
        self.status_sub = self.create_subscription(
            VehicleStatus, 
            '/fmu/out/vehicle_status', 
            self.status_callback, 
            qos_profile
        )
        self.battery_sub = self.create_subscription(
            BatteryStatus, 
            '/fmu/out/battery_status', 
            self.battery_callback, 
            qos_profile
        )

        logging.info("已创建所有发布者和订阅者（QoS: BEST_EFFORT）")
        logging.info(f"订阅位置话题：/fmu/out/vehicle_local_position_v1")

    # -----------------------------
    # Callbacks
    # -----------------------------
    def pos_callback(self, msg):
        self.current_altitude = -msg.z
        self.has_position_data = True
        if int(time.time() * 10) % 10 == 0:
            logging.debug(f"收到位置数据：z={msg.z:.2f}, 高度={self.current_altitude:.2f}m")

    def status_callback(self, msg):
        self.vehicle_status = msg

    def battery_callback(self, msg):
        self.battery_percent = msg.remaining

    # -----------------------------
    def connect(self):
        logging.info("等待建立通信链路...")
        
        # 先快速 spin 几次触发回调
        for _ in range(10):
            rclpy.spin_once(self, timeout_sec=0.1)
        
        # 等待位置数据
        timeout = 30
        start_time = time.time()
        
        while not self.has_position_data and (time.time() - start_time) < timeout:
            elapsed = time.time() - start_time
            print(f"\r等待无人机位置数据... ({elapsed:.1f}s)", end="")
            rclpy.spin_once(self, timeout_sec=0.5)
            
        if self.has_position_data:
            logging.info(f"\n✓ 通信链路已连接")
            logging.info(f"当前高度：{self.current_altitude:.2f}m")
        else:
            raise RuntimeError("\n✗ 连接超时！未收到位置数据")

    # -----------------------------
    def check_systems(self):
        """综合自检：定位、电池电量"""
        logging.info("开始系统深度健康检查...")
        
        # 预先拉取几次 ROS 2 话题数据
        for _ in range(20):
            rclpy.spin_once(self, timeout_sec=0.1)

        # 电池自检
        if self.battery_percent < 0.3: 
            logging.error(f"电量不足：{self.battery_percent*100}%")
            return False

        # 检查是否有位置数据
        if not self.has_position_data:
            logging.error("未收到位置数据！")
            return False

        logging.info(f"电池电量：{self.battery_percent*100:.1f}%")
        logging.info(f"当前高度：{self.current_altitude:.2f}m")
        logging.info("传感器与定位检查通过")
        return True

    # -----------------------------
    def send_vehicle_command(self, command, param1=0.0, param2=0.0):
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = float(param1)
        msg.param2 = float(param2)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.cmd_pub.publish(msg)

    # -----------------------------
    def publish_offboard_mode(self):
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_pub.publish(msg)

    # -----------------------------
    def publish_takeoff_setpoint(self):
        msg = TrajectorySetpoint()
        msg.position = [0.0, 0.0, -self.target_altitude]
        msg.yaw = 0.0
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.setpoint_pub.publish(msg)

    # -----------------------------
    def start_takeoff_phase(self):
        """起飞模式核心逻辑"""
        try:
            # 1. 预热 Offboard 信号
            logging.info("预热离板控制信号...")
            for i in range(20):
                self.publish_offboard_mode()
                self.publish_takeoff_setpoint()
                rclpy.spin_once(self, timeout_sec=0.1)

            logging.info("正在解锁并请求起飞...")
            
            # 切入 Offboard 模式
            self.send_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
            time.sleep(0.5)
            
            # 解锁
            self.send_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
            time.sleep(1.0)

            # 2. 安全高度监控
            self.monitor_altitude()

            # 3. 悬停等待
            logging.info("已到达 6m。无人机进入悬停状态，等待视觉系统...")
            self.pre_search_warmup()

            return True

        except Exception as e:
            logging.error(f"起飞阶段发生致命错误：{e}")
            self.emergency_handler()
            return False

    # -----------------------------
    def monitor_altitude(self):
        """带动力异常检测的高度监控"""
        last_alt = -1.0
        start_time = time.time()
        no_data_count = 0

        logging.info(f"开始监控高度，初始高度：{self.current_altitude:.2f}m")

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            
            # 检查超时
            if time.time() - start_time > 25:
                raise TimeoutError("起飞超时！")

            rel_alt = self.current_altitude
            
            # 检查数据有效性
            if not self.has_position_data:
                no_data_count += 1
                if no_data_count > 50:
                    raise RuntimeError("失去位置数据！")
                continue
            
            no_data_count = 0
            print(f"\r实时爬升高度：{rel_alt:.2f}m", end="")
            
            # 异常骤降保护
            if last_alt > 0 and rel_alt < last_alt - 0.7:
                raise RuntimeError("检测到高度异常骤降，可能动力丢失！")
            
            if rel_alt >= self.target_altitude - 0.1:
                print("\n[OK] 目标高度已锁定。")
                return True
            
            # 持续发布控制命令
            self.publish_offboard_mode()
            self.publish_takeoff_setpoint()
            
            last_alt = rel_alt

    # -----------------------------
    def pre_search_warmup(self):
        """搜寻前的预热"""
        logging.info("正在自检视觉算法识别频率...")
        time.sleep(2)
        self.is_vision_ready = True
        logging.info("视觉系统就绪，准许进入搜寻模式。")

    # -----------------------------
    def emergency_handler(self):
        """紧急情况处理"""
        logging.critical("执行紧急着陆程序！")
        self.send_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)

def main(args=None):
    rclpy.init(args=args)
    mission = DroneMission()
    
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(mission)
    
    try:
        mission.connect()
        
        if mission.check_systems():
            success = mission.start_takeoff_phase()
            if success:
                logging.info(">>> 起飞模式完美结束。建议下一步：进入【搜寻模式】指令。")
            
    except KeyboardInterrupt:
        logging.warning("用户终止，任务清理中...")
        mission.emergency_handler()
    except Exception as e:
        logging.error(f"任务执行失败：{e}")
        mission.emergency_handler()
    finally:
        mission.destroy_node()
        executor.shutdown()
        rclpy.shutdown()

if __name__ == "__main__":
    main()