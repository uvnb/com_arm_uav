#!/usr/bin/env python3
"""
NODE ROS 2: TRỰC QUAN HÓA CoM AI (1D-CNN) TRONG RVIZ2
Chức năng:
1. Đọc /joint_states thời gian thực
2. Tích lũy Sliding Window 20 frames
3. Gọi Model AI đưa ra dự báo (Predicted CoM) ở tương lai
4. Tính Toán Động Học Thực Tế (True CoM) hiện hành
5. Gửi Marker (Đỏ & Xanh) lên RViz để quan sát Phase Shift
"""

import sys
import os
import time
import numpy as np
from collections import deque

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker

import torch
import torch.nn as nn

# Chèn đường dẫn để import thư viện tự chế
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_generator import RobotTree

# ============================================================================
# CẤU HÌNH & KIẾN TRÚC MẠNG
# ============================================================================
DATADIR      = '/home/quan/robot_arm_uav/ros2_ws/preprocessed'
URDF_PATH    = '/tmp/new_arm_expanded.urdf'   # Chắc chắn xacro đã bung ra đây
WINDOW_SIZE  = 20
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ComPredictor1DCNN(nn.Module):
    def __init__(self, in_features=5, timesteps=20, out_features=2):
        super(ComPredictor1DCNN, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_features, 32, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.flatten_dim = 64 * timesteps
        self.fc_block = nn.Sequential(
            nn.Linear(self.flatten_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, out_features)
        )
    def forward(self, x):
        out = self.conv_block(x)
        out = out.view(out.size(0), -1)
        out = self.fc_block(out)
        return out


class AIVisualizerNode(Node):
    def __init__(self):
        super().__init__('ai_com_visualizer')
        
        # 1. Tải mô hình AI và Scaler
        self.get_logger().info('Đang nạp AI Model và Scaler...')
        scaler_path = os.path.join(DATADIR, 'scaler_params.npz')
        npz_scaler = np.load(scaler_path)
        # Scaler có 7 cột: [j22, j23, j28, ee_x, ee_z, com_x, com_z]
        self.mean_X = npz_scaler['mean'][:5]
        self.scale_X = npz_scaler['scale'][:5]
        self.mean_Y  = npz_scaler['mean'][5:7]
        self.scale_Y = npz_scaler['scale'][5:7]

        model_path = os.path.join(DATADIR, 'best_com_model.pth')
        self.model = ComPredictor1DCNN().to(DEVICE)
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
        self.model.eval()

        # 2. Tải cây Kinematics Tốc Độ Cao (Analytical)
        if not os.path.exists(URDF_PATH) or os.path.getsize(URDF_PATH) == 0:
            xacro_cmd = f"bash -c 'source /home/quan/robot_arm_uav/ros2_ws/install/setup.bash && xacro {os.path.dirname(os.path.abspath(__file__))}/../urdf/new_arm/new_arm.xacro > {URDF_PATH}'"
            os.system(xacro_cmd)
        self.robot_tree = RobotTree(URDF_PATH)
        
        # 3. ROS 2 Sub/Pub
        self.sub_joints = self.create_subscription(JointState, '/joint_states', self.joint_cb, 10)
        self.pub_marker = self.create_publisher(Marker, '/com_markers', 10)
        
        # Buffer cửa sổ thời gian
        self.history = deque(maxlen=WINDOW_SIZE)
        
        self.get_logger().info('✅ Node đã sẵn sàng! Mở RViz2 add Topic: /com_markers')

    def joint_cb(self, msg: JointState):
        try:
            # Map J22, J23, J28 từ joint_states
            idx_22 = msg.name.index('Revolute 22')
            idx_23 = msg.name.index('Revolute 23')
            idx_28 = msg.name.index('Revolute 28')
            j22 = msg.position[idx_22]
            j23 = msg.position[idx_23]
            j28 = msg.position[idx_28]
        except ValueError:
            return  # Đợi có đủ khớp
            
        # Tính FK bằng Analytical Tree cho bước hiện tại
        q_dict = {
            'Revolute 20': 0.0,
            'Revolute 22': j22,
            'Revolute 23': j23,
            'Revolute 26': 0.0,
            'Revolute 28': j28,
            'Revolute 30': 0.0,
            'Revolute 31': 0.0
        }
        self.robot_tree.fk_all(q_dict)
        
        # Lấy EE và Real CoM
        T_ee = self.robot_tree.links['but_1']['T_global']
        ee_x, _, ee_z = T_ee[:3, 3]
        real_com = self.robot_tree.get_com()
        
        # Lưu vào Buffer
        self.history.append([j22, j23, j28, ee_x, ee_z])
        
        # Nếu đã đủ 20 bước -> Kích hoạt AI dự báo tương lai
        if len(self.history) == WINDOW_SIZE:
            pred_com_x, pred_com_z = self.predict_future_com()
            
            # Gửi lên RViz để kiểm tra độ Mượt và Lệch pha (Phase Lead)
            self.publish_marker(real_com[0], real_com[2], is_pred=False)
            self.publish_marker(pred_com_x, pred_com_z, is_pred=True)

    def predict_future_com(self):
        # Scale Input
        raw_X = np.array(self.history)
        scaled_X = (raw_X - self.mean_X) / self.scale_X
        tensor_X = torch.tensor(scaled_X, dtype=torch.float32).transpose(0, 1).unsqueeze(0).to(DEVICE)
        
        # Inference
        with torch.no_grad():
            pred_Y_scaled = self.model(tensor_X).cpu().numpy()[0]
            
        # Giải mã kết quả Output
        pred_Y_meters = (pred_Y_scaled * self.scale_Y) + self.mean_Y
        return pred_Y_meters[0], pred_Y_meters[1]

    def publish_marker(self, x, z, is_pred):
        m = Marker()
        m.header.frame_id = "base_link"
        m.header.stamp = self.get_clock().now().to_msg()
        # id=0 là Real (Đỏ), id=1 là Pred (Xanh)
        m.id = 1 if is_pred else 0
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        
        m.pose.position.x = float(x)
        m.pose.position.y = 0.0  # Planar XZ
        m.pose.position.z = float(z)
        
        m.scale.x = 0.03
        m.scale.y = 0.03
        m.scale.z = 0.03
        
        if is_pred:
            m.color.r, m.color.g, m.color.b, m.color.a = (0.0, 1.0, 0.0, 1.0) # Xanh Lá
        else:
            m.color.r, m.color.g, m.color.b, m.color.a = (1.0, 0.0, 0.0, 1.0) # Đỏ rực
            
        self.pub_marker.publish(m)


def main():
    rclpy.init()
    node = AIVisualizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except SystemExit:
        pass
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
