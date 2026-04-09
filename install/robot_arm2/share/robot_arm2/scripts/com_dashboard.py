#!/usr/bin/env python3
"""
DASHBOARD ĐỘC LẬP: THEO DÕI TRỌNG TÂM AI 1D-CNN (Real-time)
---------------------------------------------------------
Chức năng:
1. Đọc /joint_states từ Gazebo.
2. Dự báo CoM tương lai (AI) và tính CoM hiện tại (Vật lý).
3. Đồ thị 1: Biến thiên X-CoM theo thời gian (So sánh Phase Lead).
4. Đồ thị 2: Quỹ đạo 2D X-Z (Không gian công tác).
"""

import sys
import os
import time
import numpy as np
from collections import deque
import threading

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Import RobotTree từ script cũ
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from data_generator import RobotTree
except ImportError:
    # Trường hợp lỡ tay xóa data_generator
    print("Lỗi: Không tìm thấy RobotTree trong data_generator.py")
    sys.exit(1)

# ============================================================================
# CẤU HÌNH AI
# ============================================================================
DATADIR      = '/home/quan/robot_arm_uav/ros2_ws/preprocessed'
URDF_PATH    = '/tmp/new_arm_expanded.urdf'
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
        return self.fc_block(out)

# ============================================================================
# ROS 2 NODE XỬ LÝ DỮ LIỆU
# ============================================================================
class DashboardNode(Node):
    def __init__(self):
        super().__init__('com_dashboard_node')
        
        # Tải Scaler và Model
        scaler_data = np.load(os.path.join(DATADIR, 'scaler_params.npz'))
        self.mean_X = scaler_data['mean'][:5]
        self.scale_X = scaler_data['scale'][:5]
        self.mean_Y  = scaler_data['mean'][5:7]
        self.scale_Y = scaler_data['scale'][5:7]
        
        self.model = ComPredictor1DCNN().to(DEVICE)
        self.model.load_state_dict(torch.load(os.path.join(DATADIR, 'best_com_model.pth'), 
                                              map_location=DEVICE, weights_only=True))
        self.model.eval()
        
        # Analytical Kinematics
        if not os.path.exists(URDF_PATH) or os.path.getsize(URDF_PATH) == 0:
            os.system(f"bash -c 'source /home/quan/robot_arm_uav/ros2_ws/install/setup.bash && xacro {os.path.dirname(os.path.abspath(__file__))}/../urdf/new_arm/new_arm.xacro > {URDF_PATH}'")
        self.robot_tree = RobotTree(URDF_PATH)
        
        self.sub = self.create_subscription(JointState, '/joint_states', self.joint_cb, 10)
        
        # Buffers cho Inference
        self.win_buffer = deque(maxlen=WINDOW_SIZE)
        
        # Dữ liệu để PLOT (Threads-safe-ish list)
        self.plot_time = deque(maxlen=100)
        self.plot_true_x = deque(maxlen=100)
        self.plot_pred_x = deque(maxlen=100)
        self.plot_true_z = deque(maxlen=100)
        self.plot_pred_z = deque(maxlen=100)
        
        self.start_time = time.time()
        self.get_logger().info('Dashboard Node initialized. Waiting for /joint_states...')

    def joint_cb(self, msg):
        try:
            # Map J22, 23, 28
            j_indices = [msg.name.index(name) for name in ['Revolute 22', 'Revolute 23', 'Revolute 28']]
            angles = [msg.position[i] for i in j_indices]
        except ValueError: return
        
        # Real CoM (Vật lý)
        q_dict = {'Revolute 20': 0.0, 'Revolute 22': angles[0], 'Revolute 23': angles[1], 
                  'Revolute 26': 0.0, 'Revolute 28': angles[2], 'Revolute 30': 0.0, 'Revolute 31': 0.0}
        self.robot_tree.fk_all(q_dict)
        real_com = self.robot_tree.get_com()
        
        # EE pos
        ee_pos = self.robot_tree.links['but_1']['T_global'][:3, 3]
        
        # AI Window
        self.win_buffer.append([angles[0], angles[1], angles[2], ee_pos[0], ee_pos[2]])
        
        if len(self.win_buffer) == WINDOW_SIZE:
            # Predict
            raw_X = np.array(self.win_buffer)
            scaled_X = (raw_X - self.mean_X) / self.scale_X
            tensor_X = torch.tensor(scaled_X, dtype=torch.float32).transpose(0, 1).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                pred_scaled = self.model(tensor_X).cpu().numpy()[0]
            pred_meters = (pred_scaled * self.scale_Y) + self.mean_Y
            
            # Update plot deques
            curr_t = time.time() - self.start_time
            self.plot_time.append(curr_t)
            self.plot_true_x.append(real_com[0])
            self.plot_pred_x.append(pred_meters[0])
            self.plot_true_z.append(real_com[2])
            self.plot_pred_z.append(pred_meters[1])

# ============================================================================
# MAIN GUI (Matplotlib)
# ============================================================================
def main():
    rclpy.init()
    node = DashboardNode()
    
    # Chạy ROS thread riêng
    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()
    
    # Setup Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.canvas.manager.set_window_title('AI Center of Mass Dashboard (v2)')
    
    # Subplot 1: Time Series (X-CoM)
    line_true_t, = ax1.plot([], [], 'r-', label='Real CoM_X (Physics)', linewidth=1)
    line_pred_t, = ax1.plot([], [], 'g--', label='Predicted CoM_X (AI)', linewidth=2)
    ax1.set_title('Phase Lead Analysis / Bù trễ')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Position (m)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: 2D Spatial (X-Z)
    dot_true, = ax2.plot([], [], 'ro', label='Current CoM', markersize=8)
    dot_pred, = ax2.plot([], [], 'go', label='Predicted CoM', markersize=6)
    ax2.set_title('Spatial Tracking CoM (XZ Plane)')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Z (m)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Set limit cho 2D plot (ước lượng workspace)
    ax2.set_xlim(-0.6, 0.6)
    ax2.set_ylim(-0.6, 0.6)

    def init():
        line_true_t.set_data([], [])
        line_pred_t.set_data([], [])
        dot_true.set_data([], [])
        dot_pred.set_data([], [])
        return line_true_t, line_pred_t, dot_true, dot_pred

    def update(frame):
        if len(node.plot_time) < 2: return line_true_t, line_pred_t, dot_true, dot_pred
        
        # Map time series
        t_data = list(node.plot_time)
        tx_true = list(node.plot_true_x)
        tx_pred = list(node.plot_pred_x)
        
        line_true_t.set_data(t_data, tx_true)
        line_pred_t.set_data(t_data, tx_pred)
        
        ax1.set_xlim(t_data[0], t_data[-1] + 0.1)
        # Tự động scale trục Y subplot 1
        y_min, y_max = min(tx_true + tx_pred), max(tx_true + tx_pred)
        ax1.set_ylim(y_min - 0.01, y_max + 0.01)
        
        # Map 2D spatial
        cur_rx, cur_rz = node.plot_true_x[-1], node.plot_true_z[-1]
        cur_px, cur_pz = node.plot_pred_x[-1], node.plot_pred_z[-1]
        
        dot_true.set_data([cur_rx], [cur_rz])
        dot_pred.set_data([cur_px], [cur_pz])
        
        return line_true_t, line_pred_t, dot_true, dot_pred

    ani = FuncAnimation(fig, update, init_func=init, interval=50, blit=True)
    
    plt.tight_layout()
    plt.show() # Matplotlib chặn thread chính ở đây để hiện GUI
    
    # Khi GUI tắt
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
