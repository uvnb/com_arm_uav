#!/usr/import/env python3
"""
ĐÁNH GIÁ MÔ HÌNH 1D-CNN TRÊN TẬP TEST SET BẰNG HÌNH ẢNH
Tự động xuất Báo cáo đánh giá tương tự như phiên bản tay máy ArmRobot2 cũ.
Chạy mô hình trên 300 mẫu đầu tiên của tập Test Set (Chưa từng thấy 100%).
"""

import os
import torch
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import mean_squared_error, r2_score
import torch.nn as nn

# ================================
# Cấu hình Mạng
# ================================
class ComPredictor1DCNN(nn.Module):
    def __init__(self, in_features, timesteps=30, out_features=3):
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

def main():
    DATADIR = '/home/quan/robot_arm_uav/ros2_ws/preprocessed_armfinal'
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load Scaler
    scaler_data = np.load(os.path.join(DATADIR, 'scaler_params.npz'), allow_pickle=True)
    all_mean = scaler_data['mean']
    all_scale = scaler_data['scale']
    feature_names = list(scaler_data['feature_names'])

    target_cols = ['com_x', 'com_y', 'com_z']
    target_indices = [feature_names.index(c) for c in target_cols]
    input_indices = [i for i in range(len(feature_names)) if i not in target_indices]

    mean_Y = all_mean[target_indices]
    scale_Y = all_scale[target_indices]

    # Load Model
    model = ComPredictor1DCNN(in_features=len(input_indices), out_features=len(target_cols)).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(DATADIR, 'best_com_model_armfinal.pth'), map_location=DEVICE, weights_only=True))
    model.eval()

    # Load Test Set
    test_data = np.load(os.path.join(DATADIR, 'com_preprocessed_test.npz'))
    X_test, Y_test = test_data['X'], test_data['Y']

    # Chỉ lấy 300 mẫu đầu tiên để biểu diễn đẹp như ảnh cũ
    n_samples = min(300, len(X_test))
    X_eval = X_test[:n_samples]
    Y_eval = Y_test[:n_samples]

    # Predict
    tensor_X = torch.tensor(X_eval, dtype=torch.float32).transpose(1, 2).to(DEVICE)
    with torch.no_grad():
        pred_scaled = model(tensor_X).cpu().numpy()

    # Unscale data back to meters -> millimeters
    true_mm = ((Y_eval * scale_Y) + mean_Y) * 1000
    pred_mm = ((pred_scaled * scale_Y) + mean_Y) * 1000

    tx, ty, tz = true_mm[:, 0], true_mm[:, 1], true_mm[:, 2]
    px, py, pz = pred_mm[:, 0], pred_mm[:, 1], pred_mm[:, 2]
    t = np.arange(n_samples) / 20.0  # 20Hz -> thời gian thực

    # Hàm tính metric
    def get_stats(true, pred):
        err = true - pred
        rmse = np.sqrt(mean_squared_error(true, pred))
        r2 = r2_score(true, pred)
        return err, rmse, r2

    err_x, rmse_x, r2_x = get_stats(tx, px)
    err_y, rmse_y, r2_y = get_stats(ty, py)
    err_z, rmse_z, r2_z = get_stats(tz, pz)

    print(f"--- BÁO CÁO MÔ HÌNH 3D (TEST SET N={n_samples}) ---")
    print(f"RMSE X: {rmse_x:.3f} mm | R2: {r2_x:.4f}")
    print(f"RMSE Y: {rmse_y:.3f} mm | R2: {r2_y:.4f}")
    print(f"RMSE Z: {rmse_z:.3f} mm | R2: {r2_z:.4f}")

    # ================================
    # VẼ BIỂU ĐỒ BẰNG GRIDSPEC
    # ================================
    fig = plt.figure(figsize=(16, 12))
    fig.canvas.manager.set_window_title(f"Báo cáo đánh giá (Evaluation Report)")
    fig.suptitle("Báo cáo đánh giá mô hình 1D-CNN\n(Dynamic Trajectory Dataset - Unseen Test Set)", fontsize=16, fontweight='bold')
    
    # 3 Hàng (X, Y, Z), 4 Cột (Hist, Scatter, Time[ spans 2 cols ])
    gs = gridspec.GridSpec(3, 4, figure=fig, width_ratios=[1, 1, 1, 1])

    def plot_row(row_idx, axis_name, true, pred, err, r2, rmse):
        # 1. Histogram (Cột 0)
        ax = fig.add_subplot(gs[row_idx, 0])
        ax.hist(err, bins=30, alpha=0.7, color='steelblue')
        ax.axvline(0, color='k', linestyle='--', label='Hoàn hảo')
        ax.axvline(np.mean(err), color='r', linestyle='-', label=f'Mean={np.mean(err):.2f}mm')
        ax.set_title(f'Error Histogram - {axis_name}')
        ax.set_xlabel('Sai số (mm)'); ax.set_ylabel('Tần suất')
        ax.set_xlim(-15, 15)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Scatter Actual vs Pred (Cột 1)
        ax = fig.add_subplot(gs[row_idx, 1])
        ax.scatter(true, pred, alpha=0.3, s=5, color='teal', label='Samples')
        min_v, max_v = min(min(true), min(pred)), max(max(true), max(pred))
        ax.plot([min_v, max_v], [min_v, max_v], 'k--', label='Lý tưởng')
        ax.set_title(f'Predicted vs Actual - {axis_name}\nR² = {r2:.4f}')
        ax.set_xlabel('Thực tế (mm)'); ax.set_ylabel('Dự đoán (mm)')
        ax.set_aspect('equal')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Time Series (Cột 2+3, chiếm 2 o_slot)
        ax = fig.add_subplot(gs[row_idx, 2:4])
        ax.plot(t, true, 'r-', linewidth=1.5, alpha=0.8, label='Thực tế (Ground Truth)')
        ax.plot(t, pred, 'g--', linewidth=1.5, alpha=0.8, label=f'Dự đoán AI: RMSE={rmse:.2f}mm')
        ax.fill_between(t, true, pred, color='gray', alpha=0.2, label='Vùng sai lệch')
        ax.set_title(f'So sánh theo thời gian - {axis_name} ({n_samples} mẫu đầu Test Set)')
        ax.set_xlabel('Time (s)'); ax.set_ylabel('Vị trí (mm)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    plot_row(0, 'CoM X', tx, px, err_x, r2_x, rmse_x)
    plot_row(1, 'CoM Y', ty, py, err_y, r2_y, rmse_y)
    plot_row(2, 'CoM Z', tz, pz, err_z, r2_z, rmse_z)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save Image automatically
    save_path = '/home/quan/robot_arm_uav/ros2_ws/preprocessed_armfinal/evaluation_report_3d.png'
    plt.savefig(save_path, dpi=150)
    print(f"-> Đã lưu Báo cáo dưới dạng ảnh tại: {save_path}")
    
    plt.show()

if __name__ == '__main__':
    main()
