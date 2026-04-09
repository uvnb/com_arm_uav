#!/usr/bin/env python3
"""
DYNAMIC DATA RECORDER for armfinal_description (0.5kg deployment model).
Subscribe /joint_states at 20Hz, write CSV:
  timestamp, j2, j4, j5, v2, v4, v5, ee_x, ee_z, com_x, com_z
"""

import sys
import os
import time
import csv
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

# Import RobotTree for FK/CoM
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_generator import RobotTree

URDF_PATH = '/tmp/armfinal_expanded.urdf'
CSV_PATH  = '/home/quan/robot_arm_uav/ros2_ws/armfinal_dynamic_com_dataset.csv'
RECORD_HZ = 20.0

# End-effector link name for armfinal
EE_LINK = 'bibutt_1'


class DynamicRecorderNode(Node):
    def __init__(self):
        super().__init__('dynamic_data_recorder')
        
        # Expand URDF if not exists
        if not os.path.exists(URDF_PATH) or os.path.getsize(URDF_PATH) == 0:
            xacro_cmd = (
                f"bash -c 'source /home/quan/robot_arm_uav/ros2_ws/install/setup.bash && "
                f"xacro /home/quan/robot_arm_uav/ros2_ws/src/armfinal_description/urdf/armfinal.xacro "
                f"> {URDF_PATH}'"
            )
            os.system(xacro_cmd)
        
        self.robot_tree = RobotTree(URDF_PATH)
        
        # Open CSV (append mode)
        write_header = not os.path.exists(CSV_PATH) or os.path.getsize(CSV_PATH) == 0
        self.csv_file = open(CSV_PATH, 'a', newline='')
        self.writer = csv.writer(self.csv_file)
        if write_header:
            self.writer.writerow([
                'timestamp',
                'j2', 'j4', 'j5',
                'v2', 'v4', 'v5',
                'ee_x', 'ee_y', 'ee_z',
                'com_x', 'com_y', 'com_z'
            ])
            self.csv_file.flush()
        
        self.sample_count = 0
        self.received_first_msg = False
        self.start_time = 0.0
        self.last_sample_time = 0.0
        self.sample_interval = 1.0 / RECORD_HZ
        
        # Subscribe
        self.sub = self.create_subscription(
            JointState, '/joint_states', self.joint_cb, 10
        )
        
        self.get_logger().info(
            f'📝 Data Recorder started at {RECORD_HZ}Hz for armfinal. '
            f'Saving to {CSV_PATH}. Press Ctrl+C to stop.'
        )

    def joint_cb(self, msg: JointState):
        now = time.time()
        if not self.received_first_msg:
            self.start_time = now
            self.received_first_msg = True
            self.get_logger().info('Đã nhận dữ liệu đầu tiên từ Gazebo! Bắt đầu ghi.')
            
        if (now - self.last_sample_time) < self.sample_interval:
            return
        self.last_sample_time = now
        
        try:
            idx_2 = msg.name.index('Revolute 2')
            idx_4 = msg.name.index('Revolute 4')
            idx_5 = msg.name.index('Revolute 5')
        except ValueError:
            return
        
        # Position
        j2 = msg.position[idx_2]
        j4 = msg.position[idx_4]
        j5 = msg.position[idx_5]
        
        # Velocity
        v2 = msg.velocity[idx_2] if len(msg.velocity) > idx_2 else 0.0
        v4 = msg.velocity[idx_4] if len(msg.velocity) > idx_4 else 0.0
        v5 = msg.velocity[idx_5] if len(msg.velocity) > idx_5 else 0.0
        
        # FK + CoM
        q_dict = {
            'Revolute 2': j2,
            'Revolute 4': j4,
            'Revolute 5': j5,
            'Revolute 8': 0.0,
            'Revolute 10': 0.0,
            'Revolute 12': 0.0,
        }
        self.robot_tree.fk_all(q_dict)
        
        T_ee = self.robot_tree.links[EE_LINK]['T_global']
        ee_x, ee_y, ee_z = T_ee[0, 3], T_ee[1, 3], T_ee[2, 3]
        
        com = self.robot_tree.get_com()
        
        # Write CSV
        timestamp = now - self.start_time
        self.writer.writerow([
            f"{timestamp:.4f}",
            f"{j2:.6f}", f"{j4:.6f}", f"{j5:.6f}",
            f"{v2:.6f}", f"{v4:.6f}", f"{v5:.6f}",
            f"{ee_x:.6f}", f"{ee_y:.6f}", f"{ee_z:.6f}",
            f"{com[0]:.6f}", f"{com[1]:.6f}", f"{com[2]:.6f}"
        ])
        
        self.sample_count += 1
        if self.sample_count % 200 == 0:
            self.csv_file.flush()
            elapsed = time.time() - self.start_time
            self.get_logger().info(
                f'  Đã ghi {self.sample_count} mẫu ({elapsed:.0f}s elapsed)'
            )

    def destroy_node(self):
        self.csv_file.close()
        self.get_logger().info(
            f'✅ Kết thúc ghi dữ liệu. Tổng: {self.sample_count} mẫu → {CSV_PATH}'
        )
        super().destroy_node()


def main():
    rclpy.init()
    node = DynamicRecorderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
