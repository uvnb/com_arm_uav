#!/usr/bin/env python3
"""
WORKSPACE SIMULATOR CHUYÊN NGHIỆP CHO ARMFINAL
Sử dụng Monte-Carlo Random Sampling kết hợp Động học Thuận (FK) từ URDF.
Vẽ 3 cửa sổ Figure hệt như yêu cầu:
1. Không gian 3D Scatter với Colormap theo trục Z.
2. Hình chiếu trực giao 2D (Mặt cắt Top, Side, Front).
3. Lớp màng lưới bao bọc giới hạn không gian (Convex Hull).
"""

import sys
import os
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib
matplotlib.use('TkAgg') # Môi trường GUI hiển thị cửa sổ
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Import RobotTree từ data_generator.py có sẵn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_generator import RobotTree

URDF_PATH = '/tmp/armfinal_expanded.urdf'
EE_LINK = 'bibutt_1'  # Tên Link của mút cuối (End-Effector)

def generate_workspace(num_samples=25000):
    print(f"Đang phân tích URDF và tính toán {num_samples} điểm FK. Vui lòng đợi vài giây...")
    
    # 1. Tự động sinh URDF nếu bị xóa trong thư mục tmp
    if not os.path.exists(URDF_PATH) or os.path.getsize(URDF_PATH) == 0:
        os.system(f"bash -c 'source /home/quan/robot_arm_uav/ros2_ws/install/setup.bash && xacro /home/quan/robot_arm_uav/ros2_ws/src/armfinal_description/urdf/armfinal.xacro > {URDF_PATH}'")
    
    tree = RobotTree(URDF_PATH)
    
    ee_points = []
    
    # Góc giới hạn ước chừng cho các khớp (-pi đến pi)
    for i in range(num_samples):
        # Tạo góc ngẫu nhiên cho 6 khớp
        j2 = np.random.uniform(-np.pi, np.pi)
        j4 = np.random.uniform(-np.pi, np.pi)
        j5 = np.random.uniform(-np.pi, np.pi)
        j8 = np.random.uniform(-np.pi, np.pi)
        j10 = np.random.uniform(-np.pi, np.pi)
        j12 = np.random.uniform(-np.pi, np.pi)
        
        q_dict = {
            'Revolute 2': j2, 'Revolute 4': j4, 'Revolute 5': j5,
            'Revolute 8': j8, 'Revolute 10': j10, 'Revolute 12': j12
        }
        
        # Chạy Động Học Thuận (FK)
        tree.fk_all(q_dict)
        T_ee = tree.links[EE_LINK]['T_global']
        
        # Bóc tọa độ X, Y, Z
        px, py, pz = T_ee[0, 3], T_ee[1, 3], T_ee[2, 3]
        ee_points.append([px, py, pz])
        
        if (i+1) % 5000 == 0:
            print(f"Tiến độ: {i+1}/{num_samples}")
            
    return np.array(ee_points)


def main():
    points = generate_workspace(num_samples=50000)
    X = points[:, 0]
    Y = points[:, 1]
    Z = points[:, 2]
    
    print("\nQuá trình chạy Monte-Carlo hoàn tất. Bắt đầu vẽ Biểu đồ 3-Trong-1...")
    
    fig = plt.figure(figsize=(14, 10))
    fig.canvas.manager.set_window_title('Master Workspace Simulation (With Convex Boundaries)')
    
    # Chia lưới: 2x2 siêu gọn gàng và cân đối
    gs = gridspec.GridSpec(2, 2, figure=fig)
    
    ax_3d = fig.add_subplot(gs[0, 0], projection='3d')
    ax_xy = fig.add_subplot(gs[0, 1])
    ax_xz = fig.add_subplot(gs[1, 0])
    ax_yz = fig.add_subplot(gs[1, 1])
    
    # ---------------------------------------------------------
    # 1. Vẽ Không gian 3D Scatter + Màng bao Convex Hull 3D
    # ---------------------------------------------------------
    scatter_plot = ax_3d.scatter(X, Y, Z, c=Z, cmap='jet', marker='.', s=1, alpha=0.3)
    ax_3d.scatter([0], [0], [0], color='black', s=50, marker='o', label='Base Origin')
    
    print("-> Đang bọc màng Convex Hull 3D...")
    hull3d = ConvexHull(points)
    for simplex in hull3d.simplices:
        ax_3d.plot(points[simplex, 0], points[simplex, 1], points[simplex, 2], 'r-', alpha=0.2, linewidth=0.5)

    ax_3d.set_title("Robot Workspace (3D Reachability & Boundary Envelope)", fontweight='bold')
    ax_3d.set_xlabel("X (m)"); ax_3d.set_ylabel("Y (m)"); ax_3d.set_zlabel("Z (m)")
    ax_3d.legend()
    
    cbar = plt.colorbar(scatter_plot, ax=ax_3d, shrink=0.8, pad=0.1)
    cbar.set_label('Height Z (m)')

    # Căn chỉnh khung 3D đồng tỷ lệ
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
    mid_x, mid_y, mid_z = np.mean(X), np.mean(Y), np.mean(Z)
    ax_3d.set_xlim(mid_x - max_range, mid_x + max_range)
    ax_3d.set_ylim(mid_y - max_range, mid_y + max_range)
    ax_3d.set_zlim(mid_z - max_range, mid_z + max_range)

    # ---------------------------------------------------------
    # 2. Vẽ Không gian 2D Slices + Viền Convex Hull 2D
    # ---------------------------------------------------------
    def plot_slice_with_boundary(ax, arr_x, arr_y, xl, yl, title):
        # Vẽ Scatter
        ax.scatter(arr_x, arr_y, s=0.5, color='dodgerblue', alpha=0.4, rasterized=True)
        ax.scatter([0], [0], color='black', s=30, marker='o', zorder=5) # Gốc tọa độ
        
        # Tiết chế mảng để bọc Convex Hull 2D
        pts_2d = np.column_stack((arr_x, arr_y))
        hull2d = ConvexHull(pts_2d)
        
        # Vẽ "viền" bằng cách nối các điểm thuộc vỏ lồi
        # hull.vertices trả về index các điểm ngoài cùng theo thứ tự vòng kín
        border_pts = pts_2d[hull2d.vertices, :]
        # Nối điểm cuối với điểm đầu để khép kín hình
        border_pts = np.vstack((border_pts, border_pts[0])) 
        ax.plot(border_pts[:, 0], border_pts[:, 1], 'r-', linewidth=1.5, label='Convex Boundary')
        
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    print("-> Đang bo viền Convex Hull 2D các mặt cắt...")
    plot_slice_with_boundary(ax_xy, X, Y, 'X (m)', 'Y (m)', 'Mặt cắt trên xuống (Top view: XY)')
    plot_slice_with_boundary(ax_xz, X, Z, 'X (m)', 'Z (m)', 'Mặt cắt ngang (Side view: XZ)')
    plot_slice_with_boundary(ax_yz, Y, Z, 'Y (m)', 'Z (m)', 'Mặt cắt dọc (Front view: YZ)')
    
    # Legend cho mặt cắt cuối cùng cho đẹp
    ax_yz.legend(loc='lower right', fontsize=8)

    plt.tight_layout(pad=3.0)
    print("DONE! Hệ thống đã hiển thị lên Màn hình.")
    plt.show()

if __name__ == "__main__":
    main()
