#!/usr/bin/env python3
"""
Tiền xử lý tệp armfinal_dynamic_com_dataset.csv cho mô hình 1D-CNN.
Pipeline: Lọc cột hằng → Smoothing → Scaling → Windowing → Train/Test Split

Sử dụng:
  python3 preprocess_armfinal.py
"""

import argparse
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter

# ============================================================================
# CẤU HÌNH
# ============================================================================
DEFAULT_INPUT   = '/home/quan/robot_arm_uav/ros2_ws/armfinal_dynamic_com_dataset.csv'
DEFAULT_OUTDIR  = '/home/quan/robot_arm_uav/ros2_ws/preprocessed_armfinal'

WINDOW_SIZE     = 30      # Tăng lên 30 để có "tầm nhìn" quá khứ dài hơn
FUTURE_OFFSET   = 10      # k=10 @ 20Hz = 0.5s bù trễ (MỤC TIÊU MỚI)
TRAIN_RATIO     = 0.8     # Tỷ lệ Train/Test (80/20)
SAVGOL_WINDOW   = 11      # Kích thước cửa sổ Savitzky-Golay (phải lẻ)
SAVGOL_POLY     = 3       # Bậc đa thức Savitzky-Golay

# Các cột hằng số cần loại bỏ
CONST_COLS = ['timestamp']

# Cột đầu vào (features) và cột mục tiêu (targets) cho armfinal
INPUT_FEATURES  = ['j2', 'j4', 'j5', 'v2', 'v4', 'v5', 'ee_x', 'ee_y', 'ee_z']
TARGET_FEATURES = ['com_x', 'com_y', 'com_z']


def load_and_clean(csv_path):
    """Bước 1: Đọc CSV và loại bỏ cột hằng số."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Không tìm thấy file: {csv_path}")
        
    df = pd.read_csv(csv_path)
    print(f"[1/6] Đã đọc {len(df)} mẫu từ {csv_path}")

    # Loại bỏ cột hằng số đã biết
    cols_to_drop = [c for c in CONST_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    # Tự động phát hiện thêm cột hằng số khác
    auto_const = [c for c in df.columns if df[c].std() < 1e-10]
    if auto_const:
        df = df.drop(columns=auto_const)
        cols_to_drop += auto_const

    print(f"      Loại bỏ {len(cols_to_drop)} cột hằng số: {cols_to_drop}")
    return df


def smooth_data(df, cols=None):
    """Bước 2: Làm mượt bằng Savitzky-Golay filter."""
    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()

    df_smooth = df.copy()
    n_rows = len(df)

    win = min(SAVGOL_WINDOW, n_rows)
    if win % 2 == 0: win -= 1
    if win < SAVGOL_POLY + 2:
        return df_smooth

    for col in cols:
        df_smooth[col] = savgol_filter(df[col].values, window_length=win, polyorder=SAVGOL_POLY)

    print(f"[2/6] Smoothing: window={win}, poly={SAVGOL_POLY} trên {len(cols)} cột")
    return df_smooth


def scale_data(df):
    """Bước 3: Chuẩn hóa StandardScaler."""
    scaler = StandardScaler()
    cols = df.columns.tolist()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=cols)
    print(f"[3/6] StandardScaler: {len(cols)} cột đã chuẩn hóa")
    return df_scaled, scaler


def create_windows(df, input_cols, target_cols, window_size, future_offset):
    """Bước 4 & 5: Tạo cửa sổ trượt + gán nhãn tương lai."""
    data_in  = df[input_cols].values
    data_out = df[target_cols].values
    n = len(df)

    max_idx = n - window_size - future_offset + 1
    if max_idx <= 0:
        raise ValueError(f"Dữ liệu quá ngắn! Cần ít nhất {window_size + future_offset} dòng.")

    X_list = []
    Y_list = []
    for i in range(max_idx):
        X_list.append(data_in[i : i + window_size])
        Y_list.append(data_out[i + window_size - 1 + future_offset])

    X = np.array(X_list, dtype=np.float32)
    Y = np.array(Y_list, dtype=np.float32)

    print(f"[4/6] Sliding Window: size={window_size}, future_offset={future_offset}")
    print(f"       X shape: {X.shape} | Y shape: {Y.shape}")
    return X, Y


def save_outputs(outdir, X_train, Y_train, X_test, Y_test, scaler, df_cleaned):
    """Bước 7: Lưu file .npz."""
    os.makedirs(outdir, exist_ok=True)
    np.savez_compressed(os.path.join(outdir, 'com_preprocessed_train.npz'), X=X_train, Y=Y_train)
    np.savez_compressed(os.path.join(outdir, 'com_preprocessed_test.npz'), X=X_test, Y=Y_test)
    np.savez(os.path.join(outdir, 'scaler_params.npz'),
             mean=scaler.mean_, scale=scaler.scale_,
             feature_names=np.array(df_cleaned.columns.tolist()))

    summary = (
        f"Input: {INPUT_FEATURES}\nTarget: {TARGET_FEATURES}\n"
        f"Window: {WINDOW_SIZE}, Offset: {FUTURE_OFFSET}\n"
        f"Train: {len(X_train)}, Test: {len(X_test)}\n"
    )
    with open(os.path.join(outdir, 'preprocessed_summary.txt'), 'w') as f:
        f.write(summary)
    print(f"[6/6] Đã lưu vào {outdir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default=DEFAULT_INPUT)
    parser.add_argument('--outdir', default=DEFAULT_OUTDIR)
    parser.add_argument('--window', type=int, default=WINDOW_SIZE)
    parser.add_argument('--offset', type=int, default=FUTURE_OFFSET)
    args = parser.parse_args()

    print(f"--- Tiền xử lý armfinal (0.5s Prediction) ---")
    df = load_and_clean(args.input)
    
    # Filter columns
    all_cols = INPUT_FEATURES + TARGET_FEATURES
    df = df[all_cols]

    df = smooth_data(df)
    df_scaled, scaler = scale_data(df)
    X, Y = create_windows(df_scaled, INPUT_FEATURES, TARGET_FEATURES, args.window, args.offset)
    
    # Split
    n = len(X)
    split_idx = int(n * TRAIN_RATIO)
    X_train, Y_train = X[:split_idx], Y[:split_idx]
    X_test, Y_test = X[split_idx:], Y[split_idx:]
    
    save_outputs(args.outdir, X_train, Y_train, X_test, Y_test, scaler, df)
    print("✓ Xong!")

if __name__ == '__main__':
    main()
