#!/usr/bin/env python3
"""
Huấn luyện 1D-CNN V4 cho armfinal (0.5kg model).
Mục tiêu: Bù trễ 0.5s (Future Offset = 10).
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
DATADIR      = '/home/quan/robot_arm_uav/ros2_ws/preprocessed_armfinal'
BATCH_SIZE   = 128
EPOCHS       = 100
LEARNING_RATE= 0.001
PATIENCE     = 10
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ComDataset(Dataset):
    def __init__(self, X, Y):
        # PyTorch Conv1d: (Batch, Channels, Length)
        self.X = torch.tensor(np.transpose(X, (0, 2, 1)), dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.Y[idx]

class ComPredictor1DCNN(nn.Module):
    def __init__(self, in_features, timesteps, out_features):
        super(ComPredictor1DCNN, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=in_features, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.flatten_dim = 64 * timesteps
        self.fc_block = nn.Sequential(
            nn.Linear(self.flatten_dim, 128), # Tăng layer lên 128 cho robot nhẹ
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, out_features)
        )
        
    def forward(self, x):
        out = self.conv_block(x)
        out = out.view(out.size(0), -1)
        out = self.fc_block(out)
        return out

def train_model():
    print(f"Loading data from {DATADIR}...")
    d1 = np.load(os.path.join(DATADIR, 'com_preprocessed_train.npz'))
    d2 = np.load(os.path.join(DATADIR, 'com_preprocessed_test.npz'))
    
    X_all = np.concatenate([d1['X'], d2['X']], axis=0)
    Y_all = np.concatenate([d1['Y'], d2['Y']], axis=0)
    
    full_dataset = ComDataset(X_all, Y_all)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    in_features, timesteps = full_dataset[0][0].shape
    out_features = full_dataset[0][1].shape[0]

    model = ComPredictor1DCNN(in_features, timesteps, out_features).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    
    best_val_loss = float('inf')
    best_model_path = os.path.join(DATADIR, 'best_com_model_armfinal.pth')
    epochs_no_improve = 0
    
    print("--- Training V4 (armfinal 0.5s Prediction) ---")
    for epoch in range(1, EPOCHS + 1):
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
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                out = model(bx)
                val_loss += criterion(out, by).item() * bx.size(0)
        
        avg_train_loss = train_loss / train_size
        avg_val_loss = val_loss / val_size
        
        status = ""
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            epochs_no_improve = 0
            status = "[New Best]"
        else:
            epochs_no_improve += 1
            
        print(f"Epoch {epoch:03d} | Train: {avg_train_loss:.7f} | Val: {avg_val_loss:.7f} {status}")
        if epochs_no_improve >= PATIENCE: break
            
    print(f"✓ Hoàn tất! Best Val Loss: {best_val_loss:.7f}")
    print(f"Model saved to: {best_model_path}")

if __name__ == '__main__':
    train_model()
