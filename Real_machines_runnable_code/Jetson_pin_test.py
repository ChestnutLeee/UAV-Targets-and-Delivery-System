#!/usr/bin/env python3
"""
mavros_monitor.py
通过 ROS 2 话题 /mavros/state 与 /mavros/local_position/pose
监测飞控状态与位置反馈，并评估话题发布频率及传输延迟。
解决 MAVROS Best Effort QoS 不兼容问题。
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from mavros_msgs.msg import State
from geometry_msgs.msg import PoseStamped
from collections import deque


class MavrosMonitor(Node):
    def __init__(self):
        super().__init__('mavros_monitor')

        # MAVROS 通常使用 Best Effort 发布位置话题
        maveros_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )

        self.state_sub = self.create_subscription(
            State, '/mavros/state', self.state_callback, 10)
        self.pose_sub = self.create_subscription(
            PoseStamped, '/mavros/local_position/pose', self.pose_callback, maveros_qos)

        self.pose_times = deque(maxlen=20)
        self.pose_delays = deque(maxlen=20)
        self.stats_timer = self.create_timer(1.0, self.print_stats)

        self.get_logger().info(
            'Mavros Monitor started (Best Effort QoS for pose).')

    def state_callback(self, msg: State):
        self.get_logger().info(
            f'[State] connected={msg.connected}, armed={msg.armed}, '
            f'mode={msg.mode}')

    def pose_callback(self, msg: PoseStamped):
        now = self.get_clock().now().nanoseconds * 1e-9
        self.pose_times.append(now)

        pos = msg.pose.position
        self.get_logger().info(
            f'[Pose] x={pos.x:.3f}, y={pos.y:.3f}, z={pos.z:.3f}')

        if msg.header.stamp.sec > 0:
            msg_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            delay = now - msg_time
            self.pose_delays.append(delay)

    def print_stats(self):
        freq = 0.0
        if len(self.pose_times) > 1:
            dt = self.pose_times[-1] - self.pose_times[0]
            if dt > 0:
                freq = (len(self.pose_times) - 1) / dt

        if self.pose_delays:
            avg_d = sum(self.pose_delays) / len(self.pose_delays) * 1000.0
            min_d = min(self.pose_delays) * 1000.0
            max_d = max(self.pose_delays) * 1000.0
        else:
            avg_d = min_d = max_d = 0.0

        self.get_logger().info(
            f'[Stats] Pose topic frequency: {freq:.1f} Hz | '
            f'End-to-end delay - avg: {avg_d:.1f} ms, '
            f'min: {min_d:.1f} ms, max: {max_d:.1f} ms'
        )


def main(args=None):
    rclpy.init(args=args)
    node = MavrosMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()