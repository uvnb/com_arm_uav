#!/usr/bin/env python3
"""
Tiền xử lý tệp com_dataset.csv cho mô hình 1D-CNN.
Pipeline: Lọc cột hằng → Smoothing → Scaling → Windowing → Train/Test Split

Đầu ra:
  - com_preprocessed_train.npz  (X_train, Y_train)
  - com_preprocessed_test.npz   (X_test, Y_test)
  - scaler_params.npz           (mean_, scale_ để inference sau này)
  - preprocessed_summary.txt    (thống kê quy trình)

Sử dụng:
  python3 preprocess_com.py                          # Mặc định
  python3 preprocess_com.py --input com_dataset.csv  # Chỉ định file
"""

import argparse
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter

# ============================================================================
# CẤU HÌNH (có thể chỉnh từ dòng lệnh)
# ============================================================================
DEFAULT_INPUT   = '/home/quan/robot_arm_uav/ros2_ws/com_dataset.csv'
DEFAULT_OUTDIR  = '/home/quan/robot_arm_uav/ros2_ws/preprocessed'

WINDOW_SIZE     = 20      # Kích thước cửa sổ trượt (timesteps)
FUTURE_OFFSET   = 5       # Bước "nhìn trước" (k) để bù trễ
TRAIN_RATIO     = 0.8     # Tỷ lệ Train/Test (80/20)
SAVGOL_WINDOW   = 11      # Kích thước cửa sổ Savitzky-Golay (phải lẻ)
SAVGOL_POLY     = 3       # Bậc đa thức Savitzky-Golay

# Các cột hằng số cần loại bỏ
CONST_COLS = ['ee_y', 'com_y']

# Cột đầu vào (features) và cột mục tiêu (targets)
INPUT_FEATURES  = ['j22', 'j23', 'j28', 'ee_x', 'ee_z']
TARGET_FEATURES = ['com_x', 'com_z']


def load_and_clean(csv_path):
    """Bước 1: Đọc CSV và loại bỏ cột hằng số."""
    df = pd.read_csv(csv_path)
    n_orig = len(df.columns)

    # Loại bỏ cột hằng số đã biết
    cols_to_drop = [c for c in CONST_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    # Tự động phát hiện thêm cột hằng số khác
    auto_const = [c for c in df.columns if df[c].std() < 1e-10]
    if auto_const:
        df = df.drop(columns=auto_const)
        cols_to_drop += auto_const

    print(f"[1/6] Loại bỏ {len(cols_to_drop)} cột hằng số: {cols_to_drop}")
    print(f"       Còn lại {len(df.columns)} cột: {list(df.columns)}")
    return df


def smooth_data(df, cols=None):
    """Bước 2: Làm mượt bằng Savitzky-Golay filter."""
    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()

    df_smooth = df.copy()
    n_rows = len(df)

    # Savgol yêu cầu window_length <= số dòng và phải lẻ
    win = min(SAVGOL_WINDOW, n_rows)
    if win % 2 == 0:
        win -= 1
    if win < SAVGOL_POLY + 2:
        print(f"[2/6] Bỏ qua Smoothing (dữ liệu quá ngắn: {n_rows} dòng)")
        return df_smooth

    for col in cols:
        df_smooth[col] = savgol_filter(df[col].values, window_length=win, polyorder=SAVGOL_POLY)

    print(f"[2/6] Savitzky-Golay smoothing: window={win}, poly={SAVGOL_POLY} trên {len(cols)} cột")
    return df_smooth


def scale_data(df):
    """Bước 3: Chuẩn hóa StandardScaler (mean=0, std=1)."""
    scaler = StandardScaler()
    cols = df.columns.tolist()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=cols)

    print(f"[3/6] StandardScaler: {len(cols)} cột đã chuẩn hóa")
    return df_scaled, scaler


def create_windows(df, input_cols, target_cols, window_size, future_offset):
    """
    Bước 4 & 5: Tạo cửa sổ trượt + gán nhãn tương lai.
    
    X[i] = df[input_cols] từ dòng i đến i+window_size-1
    Y[i] = df[target_cols] tại dòng i+window_size-1+future_offset
    
    Returns:
        X: shape (N_samples, window_size, n_input_features)
        Y: shape (N_samples, n_target_features)
    """
    data_in  = df[input_cols].values
    data_out = df[target_cols].values
    n = len(df)

    max_idx = n - window_size - future_offset + 1
    if max_idx <= 0:
        raise ValueError(
            f"Dữ liệu quá ngắn! Cần ít nhất {window_size + future_offset} dòng, "
            f"hiện có {n} dòng."
        )

    X_list = []
    Y_list = []
    for i in range(max_idx):
        X_list.append(data_in[i : i + window_size])
        Y_list.append(data_out[i + window_size - 1 + future_offset])

    X = np.array(X_list, dtype=np.float32)
    Y = np.array(Y_list, dtype=np.float32)

    print(f"[4/6] Sliding Window: size={window_size}, future_offset={future_offset}")
    print(f"       X shape: {X.shape}  (samples, timesteps, features)")
    print(f"       Y shape: {Y.shape}  (samples, targets)")
    return X, Y


def split_chronological(X, Y, ratio):
    """Bước 6: Chia Train/Test theo thứ tự thời gian (KHÔNG shuffle)."""
    n = len(X)
    split_idx = int(n * ratio)

    X_train, X_test = X[:split_idx], X[split_idx:]
    Y_train, Y_test = Y[:split_idx], Y[split_idx:]

    print(f"[5/6] Train/Test split: {split_idx}/{n - split_idx} "
          f"({ratio*100:.0f}%/{(1-ratio)*100:.0f}%)")
    return X_train, Y_train, X_test, Y_test


def save_outputs(outdir, X_train, Y_train, X_test, Y_test, scaler, df_cleaned):
    """Bước 7: Lưu file .npz và thống kê."""
    os.makedirs(outdir, exist_ok=True)

    np.savez_compressed(os.path.join(outdir, 'com_preprocessed_train.npz'),
                        X=X_train, Y=Y_train)
    np.savez_compressed(os.path.join(outdir, 'com_preprocessed_test.npz'),
                        X=X_test, Y=Y_test)

    # Lưu scaler params để dùng khi inference
    np.savez(os.path.join(outdir, 'scaler_params.npz'),
             mean=scaler.mean_, scale=scaler.scale_,
             feature_names=np.array(df_cleaned.columns.tolist()))

    # Báo cáo tóm tắt
    summary = (
        f"=== PREPROCESSING SUMMARY ===\n"
        f"Input features:  {INPUT_FEATURES}\n"
        f"Target features: {TARGET_FEATURES}\n"
        f"Window size:     {WINDOW_SIZE}\n"
        f"Future offset:   {FUTURE_OFFSET}\n"
        f"Smoothing:       Savitzky-Golay (w={SAVGOL_WINDOW}, p={SAVGOL_POLY})\n"
        f"Scaling:         StandardScaler\n"
        f"Train samples:   {len(X_train)}\n"
        f"Test samples:    {len(X_test)}\n"
        f"X shape:         {X_train.shape}\n"
        f"Y shape:         {Y_train.shape}\n"
    )
    with open(os.path.join(outdir, 'preprocessed_summary.txt'), 'w') as f:
        f.write(summary)

    print(f"[6/6] Đã lưu vào thư mục: {outdir}/")
    print(f"       - com_preprocessed_train.npz")
    print(f"       - com_preprocessed_test.npz")
    print(f"       - scaler_params.npz")
    print(f"       - preprocessed_summary.txt")


def main():
    parser = argparse.ArgumentParser(description='Tiền xử lý com_dataset.csv cho 1D-CNN')
    parser.add_argument('--input', default=DEFAULT_INPUT, help='Đường dẫn CSV đầu vào')
    parser.add_argument('--outdir', default=DEFAULT_OUTDIR, help='Thư mục kết quả')
    parser.add_argument('--window', type=int, default=WINDOW_SIZE, help='Kích thước cửa sổ')
    parser.add_argument('--offset', type=int, default=FUTURE_OFFSET, help='Bước nhìn trước (k)')
    parser.add_argument('--train-ratio', type=float, default=TRAIN_RATIO, help='Tỉ lệ Train')
    parser.add_argument('--no-smooth', action='store_true', help='Bỏ qua bước Smoothing')
    args = parser.parse_args()

    win_size    = args.window
    fut_offset  = args.offset
    train_ratio = args.train_ratio

    print("=" * 60)
    print("  TIỀN XỬ LÝ DỮ LIỆU CoM CHO 1D-CNN")
    print("=" * 60)
    print(f"  File đầu vào: {args.input}")
    print()

    # Pipeline
    df = load_and_clean(args.input)

    # Kiểm tra cột đầy đủ
    all_cols = INPUT_FEATURES + TARGET_FEATURES
    missing = [c for c in all_cols if c not in df.columns]
    if missing:
        print(f"LỖI: Thiếu cột {missing} trong dữ liệu!")
        return

    # Chỉ giữ cột cần dùng
    df = df[all_cols]

    if not args.no_smooth:
        df = smooth_data(df)

    df_scaled, scaler = scale_data(df)

    X, Y = create_windows(df_scaled, INPUT_FEATURES, TARGET_FEATURES,
                           win_size, fut_offset)

    X_train, Y_train, X_test, Y_test = split_chronological(X, Y, train_ratio)

    save_outputs(args.outdir, X_train, Y_train, X_test, Y_test, scaler, df)

    print()
    print("✓ HOÀN TẤT! Dữ liệu sẵn sàng cho huấn luyện 1D-CNN.")


if __name__ == '__main__':
    main()
