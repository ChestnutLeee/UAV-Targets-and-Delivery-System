#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand
from pynput import keyboard

class DroneKeyboardController(Node):
    def __init__(self):
        super().__init__('drone_keyboard_controller')
        
        # --- 控制参数 ---
        self.step_size = 0.5  # 每次按键移动的速度/距离增益
        self.target_vx, self.target_vy, self.target_vz = 0.0, 0.0, 0.0
        self.current_alt = -6.0 # 初始目标高度 (NED)

        # --- ROS 2 通信 ---
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, 
                         durability=DurabilityPolicy.TRANSIENT_LOCAL, 
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        
        self.offboard_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos)
        self.traj_pub = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos)
        self.cmd_pub = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', qos)

        # --- 键盘监听 ---
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

        # --- 定时器：持续发布指令 (20Hz) ---
        self.create_timer(0.05, self.control_loop)
        
        self.get_logger().info("\n控制说明:\n W/S: 前后 | A/D: 左右 | Q/E: 上升下降\n 请先确保仿真已启动并解锁。")

    def on_press(self, key):
        try:
            if key.char == 'w': self.target_vx = 1.0
            elif key.char == 's': self.target_vx = -1.0
            elif key.char == 'a': self.target_vy = -1.0
            elif key.char == 'd': self.target_vy = 1.0
            elif key.char == 'q': self.current_alt -= 0.2  # 向上
            elif key.char == 'e': self.current_alt += 0.2  # 向下
        except AttributeError:
            pass

    def on_release(self, key):
        # 松开按键时速度清零，实现“点动”控制
        try:
            if key.char in ['w', 's']: self.target_vx = 0.0
            if key.char in ['a', 'd']: self.target_vy = 0.0
        except AttributeError:
            pass

    def control_loop(self):
        # 1. 发布 Offboard 心跳
        off = OffboardControlMode()
        off.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        off.position, off.velocity = True, True
        self.offboard_pub.publish(off)

        # 2. 发布控制期望值
        traj = TrajectorySetpoint()
        traj.timestamp = off.timestamp
        # 混合控制：XY使用速度，Z使用位置(高度锁定)
        traj.velocity = [float(self.target_vx), float(self.target_vy), float('nan')]
        traj.position = [float('nan'), float('nan'), float(self.current_alt)]
        traj.yaw = 0.0
        self.traj_pub.publish(traj)

    def send_command(self, command, p1=0.0, p2=0.0):
        msg = VehicleCommand()
        msg.command, msg.param1, msg.param2 = command, p1, p2
        msg.target_system, msg.target_component = 1, 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.cmd_pub.publish(msg)

def main():
    rclpy.init()
    node = DroneKeyboardController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()