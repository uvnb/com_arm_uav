#!/usr/bin/env python3
"""
DASHBOARD V4: THEO DÕI TRỌNG TÂM AI 1D-CNN CHO armfinal (0.5kg)
Mục tiêu: Bù trễ 0.5s (Future Prediction)
-----------------------------------------------------------------------------
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
from std_msgs.msg import Float64

import torch
import torch.nn as nn
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation

# Import RobotTree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_generator import RobotTree

# ============================================================================
# CẤU HÌNH
# ============================================================================
DATADIR      = '/home/quan/robot_arm_uav/ros2_ws/preprocessed_armfinal'
URDF_PATH    = '/tmp/armfinal_expanded.urdf'
WINDOW_SIZE  = 30
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EE_LINK      = 'bibutt_1'


class ComPredictor1DCNN(nn.Module):
    def __init__(self, in_features, timesteps=30, out_features=2):
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
            nn.Linear(self.flatten_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, out_features)
        )
    def forward(self, x):
        out = self.conv_block(x)
        out = out.view(out.size(0), -1)
        return self.fc_block(out)


class DashboardNode(Node):
    def __init__(self):
        super().__init__('com_dashboard_node_armfinal')
        
        # Tải Scaler
        scaler_path = os.path.join(DATADIR, 'scaler_params.npz')
        if not os.path.exists(scaler_path):
             self.get_logger().error(f"Missing scaler: {scaler_path}")
             sys.exit(1)
             
        scaler_data = np.load(scaler_path, allow_pickle=True)
        all_mean  = scaler_data['mean']
        all_scale = scaler_data['scale']
        feature_names = list(scaler_data['feature_names'])
        
        target_cols = ['com_x', 'com_y', 'com_z']
        target_indices = [feature_names.index(c) for c in target_cols]
        input_indices  = [i for i in range(len(feature_names)) if i not in target_indices]
        
        self.mean_X  = all_mean[input_indices]
        self.scale_X = all_scale[input_indices]
        self.mean_Y  = all_mean[target_indices]
        self.scale_Y = all_scale[target_indices]
        self.n_features = len(input_indices)
        
        # Tải Model
        self.model = ComPredictor1DCNN(in_features=self.n_features, out_features=len(target_indices)).to(DEVICE)
        model_path = os.path.join(DATADIR, 'best_com_model_armfinal.pth')
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
        self.model.eval()
        
        # URDF Tree
        if not os.path.exists(URDF_PATH) or os.path.getsize(URDF_PATH) == 0:
            os.system(f"bash -c 'source /home/quan/robot_arm_uav/ros2_ws/install/setup.bash && xacro /home/quan/robot_arm_uav/ros2_ws/src/armfinal_description/urdf/armfinal.xacro > {URDF_PATH}'")
        self.robot_tree = RobotTree(URDF_PATH)
        
        self.sub = self.create_subscription(JointState, '/joint_states', self.joint_cb, 10)
        self.pub_pred_x = self.create_publisher(Float64, '/ai_pred_com_x', 10)
        
        self.win_buffer = deque(maxlen=WINDOW_SIZE)
        self.plot_time   = deque(maxlen=300)
        self.plot_true_x = deque(maxlen=300)
        self.plot_pred_x = deque(maxlen=300)
        self.plot_true_y = deque(maxlen=300)
        self.plot_pred_y = deque(maxlen=300)
        self.plot_true_z = deque(maxlen=300)
        self.plot_pred_z = deque(maxlen=300)
        self.start_time = time.time()
        
        self.get_logger().info('✅ armfinal Dashboard sẵn sàng (Mục tiêu 0.5s)!')

    def joint_cb(self, msg):
        try:
            idx_2 = msg.name.index('Revolute 2')
            idx_4 = msg.name.index('Revolute 4')
            idx_5 = msg.name.index('Revolute 5')
        except ValueError:
            return
        
        j2, j4, j5 = msg.position[idx_2], msg.position[idx_4], msg.position[idx_5]
        
        # FK
        q_dict = {'Revolute 2': j2, 'Revolute 4': j4, 'Revolute 5': j5, 
                  'Revolute 8': 0.0, 'Revolute 10': 0.0, 'Revolute 12': 0.0}
        self.robot_tree.fk_all(q_dict)
        real_com = self.robot_tree.get_com()
        T_ee = self.robot_tree.links[EE_LINK]['T_global']
        ee_x, ee_y, ee_z = T_ee[0, 3], T_ee[1, 3], T_ee[2, 3]
        
        v2 = msg.velocity[idx_2] if len(msg.velocity) > idx_2 else 0.0
        v4 = msg.velocity[idx_4] if len(msg.velocity) > idx_4 else 0.0
        v5 = msg.velocity[idx_5] if len(msg.velocity) > idx_5 else 0.0
        
        row = [j2, j4, j5, v2, v4, v5, ee_x, ee_y, ee_z]
        self.win_buffer.append(row)
        
        if len(self.win_buffer) == WINDOW_SIZE:
            px, py, pz = self._predict()
            
            # --- PUBLIC OUPUT FOR BATTERY BALANCER ---
            msg_x = Float64()
            msg_x.data = float(px)
            self.pub_pred_x.publish(msg_x)
            
            curr_t = time.time() - self.start_time
            self.plot_time.append(curr_t)
            self.plot_true_x.append(real_com[0])
            self.plot_pred_x.append(px)
            self.plot_true_y.append(real_com[1])
            self.plot_pred_y.append(py)
            self.plot_true_z.append(real_com[2])
            self.plot_pred_z.append(pz)

    def _predict(self):
        raw_X = np.array(self.win_buffer)
        scaled_X = (raw_X - self.mean_X) / self.scale_X
        # Torch Conv1d: (Batch, Channels, Length)
        tensor_X = torch.tensor(scaled_X, dtype=torch.float32).transpose(0, 1).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred_scaled = self.model(tensor_X).cpu().numpy()[0]
        pred_meters = (pred_scaled * self.scale_Y) + self.mean_Y
        return pred_meters[0], pred_meters[1], pred_meters[2]


def main():
    rclpy.init()
    node = DashboardNode()
    
    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()
    
    fig = plt.figure(figsize=(16, 9))
    fig.canvas.manager.set_window_title('V5 AI 3D Dashboard (armfinal - 0.5s Horizon)')
    gs = gridspec.GridSpec(3, 3, figure=fig, width_ratios=[1.2, 1, 1.5])
    
    # Cột 0: 3 Biểu đồ Time-Series
    ax_tx = fig.add_subplot(gs[0, 0])
    ax_ty = fig.add_subplot(gs[1, 0])
    ax_tz = fig.add_subplot(gs[2, 0])
    
    # Cột 1: 3 Biểu đồ Không gian 2D (Oxz, Oyz, Oxy)
    ax_xz = fig.add_subplot(gs[0, 1])
    ax_yz = fig.add_subplot(gs[1, 1])
    ax_xy = fig.add_subplot(gs[2, 1])
    
    # Cột 2: Khối 3D Projection (Kéo dài cả 3 hàng)
    ax_3d = fig.add_subplot(gs[:, 2], projection='3d')
    
    # --- Cài đặt Time-Series ---
    line_tx, = ax_tx.plot([], [], 'r-', label='Real X')
    line_px, = ax_tx.plot([], [], 'g--', label='AI Pred X')
    ax_tx.set_ylabel('CoM X (m)'); ax_tx.grid(True, alpha=0.3)
    
    line_ty, = ax_ty.plot([], [], 'r-', label='Real Y')
    line_py, = ax_ty.plot([], [], 'g--', label='AI Pred Y')
    ax_ty.set_ylabel('CoM Y (m)'); ax_ty.grid(True, alpha=0.3)
    
    line_tz, = ax_tz.plot([], [], 'r-', label='Real Z')
    line_pz, = ax_tz.plot([], [], 'g--', label='AI Pred Z')
    ax_tz.set_ylabel('CoM Z (m)'); ax_tz.set_xlabel('Time (s)'); ax_tz.grid(True, alpha=0.3)

    # --- Cài đặt 2D Spatial ---
    def setup_2d(ax, title, xl, yl):
        dt, = ax.plot([], [], 'ro')
        dp, = ax.plot([], [], 'g^')
        tt, = ax.plot([], [], 'r-', alpha=0.3)
        tp, = ax.plot([], [], 'g-', alpha=0.3)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(xl, fontsize=8); ax.set_ylabel(yl, fontsize=8)
        ax.grid(True, alpha=0.3)
        return dt, dp, tt, tp

    dt_xz, dp_xz, tt_xz, tp_xz = setup_2d(ax_xz, 'Oxz Plane (X vs Z)', 'X (m)', 'Z (m)')
    dt_yz, dp_yz, tt_yz, tp_yz = setup_2d(ax_yz, 'Oyz Plane (Y vs Z)', 'Y (m)', 'Z (m)')
    dt_xy, dp_xy, tt_xy, tp_xy = setup_2d(ax_xy, 'Oxy Plane (X vs Y)', 'X (m)', 'Y (m)')

    # --- Cài đặt 3D Spatial ---
    dot_true_3d, = ax_3d.plot([], [], [], 'ro', label='Current CoM')
    dot_pred_3d, = ax_3d.plot([], [], [], 'g^', label='AI Future CoM')
    trail_true_3d, = ax_3d.plot([], [], [], 'r-', alpha=0.3)
    trail_pred_3d, = ax_3d.plot([], [], [], 'g-', alpha=0.3)
    
    ax_3d.set_title('3D Spatial Tracking')
    ax_3d.set_xlabel('X (m)'); ax_3d.set_ylabel('Y (m)'); ax_3d.set_zlabel('Z (m)')
    ax_3d.legend(loc='upper right')
    
    # Origin Triad
    ax_3d.plot([-0.05, 0.05], [0, 0], [0, 0], 'k-', alpha=0.2)
    ax_3d.plot([0, 0], [-0.05, 0.05], [0, 0], 'k-', alpha=0.2)
    ax_3d.plot([0, 0], [0, 0], [0, 0.5], 'k-', alpha=0.2)

    text_metrics = ax_tx.text(0.02, 0.90, '', transform=ax_tx.transAxes, 
                            verticalalignment='top', fontsize=10, 
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    def init():
        return (line_tx, line_px, line_ty, line_py, line_tz, line_pz, 
                dt_xz, dp_xz, tt_xz, tp_xz, dt_yz, dp_yz, tt_yz, tp_yz, dt_xy, dp_xy, tt_xy, tp_xy, 
                dot_true_3d, dot_pred_3d, trail_true_3d, trail_pred_3d, text_metrics)

    def update(frame):
        if len(node.plot_time) < 2:
            return (line_tx, line_px, line_ty, line_py, line_tz, line_pz, 
                dt_xz, dp_xz, tt_xz, tp_xz, dt_yz, dp_yz, tt_yz, tp_yz, dt_xy, dp_xy, tt_xy, tp_xy, 
                dot_true_3d, dot_pred_3d, trail_true_3d, trail_pred_3d, text_metrics)
        
        t = list(node.plot_time)
        tx, px = list(node.plot_true_x), list(node.plot_pred_x)
        ty, py = list(node.plot_true_y), list(node.plot_pred_y)
        tz, pz = list(node.plot_true_z), list(node.plot_pred_z)
        
        err_x = np.array(tx) - np.array(px)
        err_y = np.array(ty) - np.array(py)
        err_z = np.array(tz) - np.array(pz)
        dist_err = np.sqrt(err_x**2 + err_y**2 + err_z**2) * 1000 
        
        live_err = dist_err[-1]
        avg_err = np.mean(dist_err)

        # Cập nhật Time-series
        line_tx.set_data(t, tx); line_px.set_data(t, px)
        line_ty.set_data(t, ty); line_py.set_data(t, py)
        line_tz.set_data(t, tz); line_pz.set_data(t, pz)
        
        for ax, arr1, arr2 in zip([ax_tx, ax_ty, ax_tz], [tx, ty, tz], [px, py, pz]):
            ax.set_xlim(t[0], t[-1] + 1.0)
            combo = arr1 + arr2
            ax.set_ylim(min(combo) - 0.01, max(combo) + 0.01)

        # Cập nhật 2D Spatial
        dr = 50 # trail length
        dt_xz.set_data([tx[-1]], [tz[-1]]); dp_xz.set_data([px[-1]], [pz[-1]])
        tt_xz.set_data(tx[-dr:], tz[-dr:]); tp_xz.set_data(px[-dr:], pz[-dr:])
        
        dt_yz.set_data([ty[-1]], [tz[-1]]); dp_yz.set_data([py[-1]], [pz[-1]])
        tt_yz.set_data(ty[-dr:], tz[-dr:]); tp_yz.set_data(py[-dr:], pz[-dr:])
        
        dt_xy.set_data([tx[-1]], [ty[-1]]); dp_xy.set_data([px[-1]], [py[-1]])
        tt_xy.set_data(tx[-dr:], ty[-dr:]); tp_xy.set_data(px[-dr:], py[-dr:])
        
        for ax, arx1, ary1, arx2, ary2 in zip(
            [ax_xz, ax_yz, ax_xy], 
            [tx, ty, tx], [tz, tz, ty], 
            [px, py, px], [pz, pz, py]):
            all_x = arx1[-dr:] + arx2[-dr:]
            all_y = ary1[-dr:] + ary2[-dr:]
            ax.set_xlim(min(all_x)-0.01, max(all_x)+0.01)
            ax.set_ylim(min(all_y)-0.01, max(all_y)+0.01)

        # Cập nhật 3D Spatial
        dot_true_3d.set_data_3d([tx[-1]], [ty[-1]], [tz[-1]])
        dot_pred_3d.set_data_3d([px[-1]], [py[-1]], [pz[-1]])
        trail_true_3d.set_data_3d(tx[-dr:], ty[-dr:], tz[-dr:])
        trail_pred_3d.set_data_3d(px[-dr:], py[-dr:], pz[-dr:])
        
        ax_3d.set_xlim(min(tx[-dr:] + px[-dr:]) - 0.01, max(tx[-dr:] + px[-dr:]) + 0.01)
        ax_3d.set_ylim(min(ty[-dr:] + py[-dr:]) - 0.01, max(ty[-dr:] + py[-dr:]) + 0.01)
        ax_3d.set_zlim(min(tz[-dr:] + pz[-dr:]) - 0.01, max(tz[-dr:] + pz[-dr:]) + 0.01)
            
        text_metrics.set_text(f'3D SPATIAL ERROR\nLive: {live_err:.1f} mm\nAvg: {avg_err:.1f} mm')
        
        return (line_tx, line_px, line_ty, line_py, line_tz, line_pz, 
                dt_xz, dp_xz, tt_xz, tp_xz, dt_yz, dp_yz, tt_yz, tp_yz, dt_xy, dp_xy, tt_xy, tp_xy, 
                dot_true_3d, dot_pred_3d, trail_true_3d, trail_pred_3d, text_metrics)

    ani = FuncAnimation(fig, update, init_func=init, interval=50, blit=True)
    plt.tight_layout()
    plt.show()
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
