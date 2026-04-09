#!/usr/bin/env python3
"""
ROBOT DANCE: Điều khiển cánh tay armfinal "nhảy múa" trong Ignition Gazebo.
Phát quỹ đạo Sin/Cos với biên độ và tần số đa dạng cho J2, J4, J5.
Mỗi "bài nhảy" kéo dài ~60 giây, tổng cộng 10 bài = 10 phút dữ liệu.

JOINTS: Revolute 2 (Base), Revolute 4 (Shoulder), Revolute 5 (Elbow)
LOCKED: Revolute 8, 10, 12 (Wrist) = 0
"""

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import numpy as np
import time

# Joint limits (radian) from armfinal.ros2_control.xacro
JOINT_LIMITS = {
    'Revolute 2': (-3.141593, 3.141593),
    'Revolute 4': (-1.570796, 1.570796),
    'Revolute 5': (-1.570796, 1.570796),
}

# All 6 joints (wrist joints locked at 0)
ALL_JOINTS = [
    'Revolute 2', 'Revolute 4', 'Revolute 5',
    'Revolute 8', 'Revolute 10', 'Revolute 12'
]

# Dance routines: (freq_J2, freq_J4, freq_J5, amp_scale, phase_offset, duration_s)
DANCE_ROUTINES = [
    (0.3, 0.2, 0.4, 0.3, 0.0,   30),   # Slow warmup
    (0.2, 0.15, 0.25, 0.8, 0.5, 30),   # Slow, large amplitude
    (0.5, 0.4, 0.6, 0.6, 1.0,   30),   # Medium, multi-phase
    (1.0, 0.8, 1.2, 0.3, 0.0,   30),   # Fast, small amplitude
    (0.8, 0.6, 1.0, 0.7, 1.5,   30),   # Fast, large amplitude
    (0.3, 0.7, 0.5, 0.5, 2.0,   30),   # Mixed frequency
    (0.1, 0.08, 0.12, 0.9, 0.0, 30),   # Ultra-slow (large inertia)
    (0.4, 0.3, 0.5, 0.6, 0.7,   30),   # Gradual acceleration
    (0.6, 0.9, 0.4, 0.5, 3.14,  30),   # Step-like sin
    (0.5, 0.5, 0.5, 0.7, 1.57,  30),   # Sin+Cos phase offset
]


class DanceTrajectoryNode(Node):
    def __init__(self):
        super().__init__('dance_trajectory_node')
        self.pub = self.create_publisher(
            JointTrajectory,
            '/arm_controller/joint_trajectory',
            10
        )
        self.get_logger().info('Dance Trajectory Node started for armfinal. Robot sẽ bắt đầu nhảy...')

    def compute_joint_angle(self, t, freq, amp_scale, phase, low, high):
        mid = (low + high) / 2.0
        half_range = (high - low) / 2.0
        angle = mid + amp_scale * half_range * np.sin(2 * np.pi * freq * t + phase)
        return np.clip(angle, low, high)

    def send_position(self, positions):
        msg = JointTrajectory()
        msg.joint_names = ALL_JOINTS
        
        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start = Duration(sec=0, nanosec=100_000_000)  # 0.1s
        
        msg.points = [point]
        self.pub.publish(msg)

    def run_dance(self, routine_idx, freq_j2, freq_j4, freq_j5, amp, phase, duration):
        self.get_logger().info(
            f'▶ Bài nhảy {routine_idx+1}/{len(DANCE_ROUTINES)}: '
            f'freq=({freq_j2:.1f},{freq_j4:.1f},{freq_j5:.1f}) '
            f'amp={amp:.1f} duration={duration}s'
        )
        
        dt = 0.05  # 20Hz
        t = 0.0
        lim = JOINT_LIMITS
        
        while t < duration:
            j2 = self.compute_joint_angle(t, freq_j2, amp, phase, *lim['Revolute 2'])
            j4 = self.compute_joint_angle(t, freq_j4, amp, phase * 0.7, *lim['Revolute 4'])
            j5 = self.compute_joint_angle(t, freq_j5, amp, phase * 1.3, *lim['Revolute 5'])
            
            # Revolute 8, 10, 12 = 0 (locked wrist)
            positions = [float(j2), float(j4), float(j5), 0.0, 0.0, 0.0]
            self.send_position(positions)
            
            time.sleep(dt)
            t += dt
            
            rclpy.spin_once(self, timeout_sec=0.001)


def main():
    rclpy.init()
    node = DanceTrajectoryNode()
    
    total_routines = len(DANCE_ROUTINES)
    total_time = sum(r[5] for r in DANCE_ROUTINES)
    node.get_logger().info(f'Tổng cộng {total_routines} bài nhảy, ~{total_time/60:.0f} phút.')
    
    try:
        # Wait for Gazebo
        node.get_logger().info('Đang chờ Gazebo khởi động...')
        while rclpy.ok() and node.pub.get_subscription_count() == 0:
            rclpy.spin_once(node, timeout_sec=0.5)
        node.get_logger().info('Đã kết nối với Gazebo! Bắt đầu phát dữ liệu...')
        
        for i, (f2, f4, f5, amp, phase, dur) in enumerate(DANCE_ROUTINES):
            node.run_dance(i, f2, f4, f5, amp, phase, dur)
            time.sleep(2.0)
        
        node.get_logger().info('✅ Hoàn tất tất cả các bài nhảy!')
    except KeyboardInterrupt:
        node.get_logger().info('⏹ Dừng bởi người dùng (Ctrl+C).')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
