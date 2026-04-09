#!/usr/bin/env python3
"""
Script dự đoán thời gian thực (Inference) sử dụng Mô hình 1D-CNN.

Quy trình:
1. Đọc dữ liệu thô (chưa scale).
2. Tự tạo Window 20 bước.
3. Chạy qua StandardScaler lưu từ trước.
4. Dự báo với bộ não AI best_com_model.pth.
5. Giải mã ngược (Inverse Scale) về gốc tọa độ Mét.
6. So sánh trực tiếp với tọa độ CoM vật lý (thực tế) để tính sai số Milimet.
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ============================================================================
# CẤU HÌNH VÀ KIẾN TRÚC MẠNG
# ============================================================================
DATADIR      = '/home/quan/robot_arm_uav/ros2_ws/preprocessed'
RAW_CSV      = '/home/quan/robot_arm_uav/ros2_ws/com_dataset.csv'
WINDOW_SIZE  = 20
FUTURE_OFFSET= 5
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

def main():
    print("=" * 60)
    print(" INFERENCE: KIỂM TRA ĐỘ CHÍNH XÁC CỦA BỘ NÃO AI 1D-CNN")
    print("=" * 60)
    
    # 1. Tải mảng Scale Parameters
    scaler_path = os.path.join(DATADIR, 'scaler_params.npz')
    if not os.path.exists(scaler_path):
        print("Lỗi: Không tìm thấy scaler_params.npz")
        return
        
    npz_scaler = np.load(scaler_path)
    # Scaler có 7 cột: [j22, j23, j28, ee_x, ee_z, com_x, com_z]
    mean_full = npz_scaler['mean']
    scale_full= npz_scaler['scale']
    
    mean_X = mean_full[:5]
    scale_X= scale_full[:5]
    mean_Y = mean_full[5:7]
    scale_Y= scale_full[5:7]
    
    # 2. Tải Khối lượng trí tuệ nhân tạo (Weights)
    model_path = os.path.join(DATADIR, 'best_com_model.pth')
    model = ComPredictor1DCNN(in_features=5, timesteps=WINDOW_SIZE, out_features=2)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()  # Chuyển về chế độ Evaluate (TẮT DROPOUT!)
    print("✓ Đã nạp thành công bộ trọng số siêu việt best_com_model.pth")
    
    # 3. Đọc dữ liệu vật lý chưa qua xử lý làm bài Test
    print("Đang nạp file CSV gốc để mô phỏng dữ liệu thời gian thực...")
    df = pd.read_csv(RAW_CSV)
    
    input_cols = ['j22', 'j23', 'j28', 'ee_x', 'ee_z']
    target_cols = ['com_x', 'com_z']
    
    max_idx = len(df) - WINDOW_SIZE - FUTURE_OFFSET
    
    print("\n--- BẮT ĐẦU CHẠY THỬ NGHIỆM INFERENCE ---")
    num_tests = 5
    
    # Chọn ngẫu nhiên 5 thời điểm khác nhau chưa từng nằm sát nhau
    test_indices = random.sample(range(0, max_idx), num_tests)
    
    total_error_x_mm = 0.0
    total_error_z_mm = 0.0
    
    for i, start_idx in enumerate(test_indices, 1):
        # Lấy một nùi thời gian (Window) 20 bước hiện tại
        window_df = df[input_cols].iloc[start_idx : start_idx + WINDOW_SIZE]
        raw_X = window_df.values
        
        # 4. Tiền xử lý (Scaling) chính xác như Data Pipeline
        scaled_X = (raw_X - mean_X) / scale_X
        
        # Chuyển dạng (Batch=1, Features=5, Timesteps=20)
        tensor_X = torch.tensor(scaled_X, dtype=torch.float32).transpose(0, 1).unsqueeze(0).to(DEVICE)
        
        # 5. DỰ ĐOÁN
        with torch.no_grad():
            pred_Y_scaled = model(tensor_X).cpu().numpy()[0]
            
        # 6. Giải mã ngược về Mét (Inverse Scale)
        pred_Y_meters = (pred_Y_scaled * scale_Y) + mean_Y
        pred_com_x, pred_com_z = pred_Y_meters
        
        # 7. So sánh với Giá trị Thật sự K ở tương lai
        future_idx = start_idx + WINDOW_SIZE - 1 + FUTURE_OFFSET
        truth_com_x = df['com_x'].iloc[future_idx]
        truth_com_z = df['com_z'].iloc[future_idx]
        
        # Tính sai số theo Milimet
        err_x_mm = abs(pred_com_x - truth_com_x) * 1000.0
        err_z_mm = abs(pred_com_z - truth_com_z) * 1000.0
        
        total_error_x_mm += err_x_mm
        total_error_z_mm += err_z_mm
        
        print(f"\n[Test {i}] Dữ liệu thu vào ở Offset {start_idx} (Đoạn 20 steps)")
        print(f"  Thực tế (Vật lý) ở t+{FUTURE_OFFSET}: CoM_X = {truth_com_x:8.5f} m, CoM_Z = {truth_com_z:8.5f} m")
        print(f"  AI Dự đoán         ở t+{FUTURE_OFFSET}: CoM_X = {pred_com_x:8.5f} m, CoM_Z = {pred_com_z:8.5f} m")
        print(f"  -> Lệch X: {err_x_mm:.2f} mm | Lệch Z: {err_z_mm:.2f} mm")

    print("\n" + "="*50)
    print(f"Tổng kết độ chính xác Inference sau {num_tests} phép thử:")
    print(f"Trung bình sai số Trục X: {total_error_x_mm / num_tests:.3f} mm")
    print(f"Trung bình sai số Trục Z: {total_error_z_mm / num_tests:.3f} mm")
    print("="*50)
    
if __name__ == '__main__':
    main()
