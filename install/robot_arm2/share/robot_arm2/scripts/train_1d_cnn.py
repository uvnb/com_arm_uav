#!/usr/bin/env python3
"""
Huấn luyện mô hình 1D-CNN NÂNG CẤP (v2).
Cải tiến:
  - Thêm Dropout (0.3) chống overfitting.
  - Xáo trộn dữ liệu (Shuffle) để bao phủ toàn bộ workspace.
  - Early Stopping: Dừng sớm nếu Val Loss không cải thiện.
  - L2 Regularization (Weight Decay).
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

# ============================================================================
# CẤU HÌNH
# ============================================================================
DATADIR      = '/home/quan/robot_arm_uav/ros2_ws/preprocessed'
BATCH_SIZE   = 128
EPOCHS       = 100        # Tăng Epoch nhưng sẽ dùng EarlyStopping
LEARNING_RATE= 0.001
PATIENCE     = 8          # Nếu sau 8 epoch Val Loss không giảm thì dừng
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# DATALOADER TÙY CHỈNH
# ============================================================================
class ComDataset(Dataset):
    def __init__(self, X, Y):
        # PyTorch Conv1d: (Batch, Channels, Length) -> (Batch, 5, 20)
        self.X = torch.tensor(np.transpose(X, (0, 2, 1)), dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# ============================================================================
# KIẾN TRÚC MẠNG 1D-CNN (NÂNG CẤP)
# ============================================================================
class ComPredictor1DCNN(nn.Module):
    def __init__(self, in_features, timesteps, out_features):
        super(ComPredictor1DCNN, self).__init__()
        
        # Convolution blocks
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=in_features, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2), # Dropout lớp convolution nhẹ
            
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.flatten_dim = 64 * timesteps
        
        # Dense layers
        self.fc_block = nn.Sequential(
            nn.Linear(self.flatten_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3), # Dropout lớp dense mạnh hơn
            
            nn.Linear(64, out_features)
        )
        
    def forward(self, x):
        out = self.conv_block(x)
        out = out.view(out.size(0), -1)
        out = self.fc_block(out)
        return out

# ============================================================================
# VÒNG LẶP HUẤN LUYỆN
# ============================================================================
def train_model():
    # 1. Load và GỘP dữ liệu để xáo trộn (Tránh Distribution Shift của Cartesian Grid)
    print("Đang tải và chuẩn bị dữ liệu (Gộp & Xáo trộn)...")
    train_file = os.path.join(DATADIR, 'com_preprocessed_train.npz')
    test_file  = os.path.join(DATADIR, 'com_preprocessed_test.npz')
    
    d1 = np.load(train_file)
    d2 = np.load(test_file)
    
    X_all = np.concatenate([d1['X'], d2['X']], axis=0)
    Y_all = np.concatenate([d1['Y'], d2['Y']], axis=0)
    
    full_dataset = ComDataset(X_all, Y_all)
    
    # Chia lại 80/20 với Shuffling ngẫu nhiên
    train_size = int(0.8 * len(full_dataset))
    val_size   = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # Thông số mạng
    in_features, timesteps = full_dataset[0][0].shape
    out_features = full_dataset[0][1].shape[0]

    print(f"Dataset Size: {len(full_dataset)} | Train: {train_size} | Val: {val_size}")
    print(f"Thiết bị: {DEVICE}\n")

    model = ComPredictor1DCNN(in_features, timesteps, out_features).to(DEVICE)
    criterion = nn.MSELoss()
    # Thêm Weight Decay (L2 Regulation) chống overfitting
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    
    best_val_loss = float('inf')
    best_model_path = os.path.join(DATADIR, 'best_com_model.pth')
    
    # Biến phục vụ Early Stopping
    epochs_no_improve = 0
    
    print("================ BẮT ĐẦU HUẤN LUYỆN (V2) ================")
    start_time = time.time()
    
    for epoch in range(1, EPOCHS + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * bx.size(0)
        
        avg_train_loss = train_loss / train_size
        
        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                out = model(bx)
                val_loss += criterion(out, by).item() * bx.size(0)
        
        avg_val_loss = val_loss / val_size
        
        # --- Log Progress ---
        status = ""
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            epochs_no_improve = 0
            status = " [MỚI]"
        else:
            epochs_no_improve += 1
            status = f" ({epochs_no_improve}/{PATIENCE})"
            
        print(f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.7f} | Val Loss: {avg_val_loss:.7f}{status}")
        
        # --- Early Stopping check ---
        if epochs_no_improve >= PATIENCE:
            print(f"\n[Dừng sớm] Không có cải thiện sau {PATIENCE} epochs.")
            break
            
    total_time = (time.time() - start_time) / 60
    print(f"\n✓ Hoàn tất sau {total_time:.2f} phút.")
    print(f"Best Val Loss (MSE): {best_val_loss:.7f}")
    print(f"Trọng số lưu tại: {best_model_path}")

if __name__ == '__main__':
    train_model()
