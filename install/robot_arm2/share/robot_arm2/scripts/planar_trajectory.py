#!/usr/bin/env python3
"""
Điều khiển quỹ đạo thẳng đứng cho cánh tay robot trong mặt phẳng XZ (y=0).
Sử dụng Scipy SLSQP IK Solver (khóa cứng J20, J26, J30, J31 ở 0).
"""
import sys
import os
import math
import numpy as np
from scipy.optimize import minimize
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

# Import FK từ file fk.py cùng thư mục
try:
    from fk import fk
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from fk import fk

def ik_planar_scipy(x_target, z_target, q_init=(0.0, 0.0, 0.0)):
    """
    Sử dụng scipy.optimize để tìm J22, J23, J28 (ở radian)
    sao cho End-effector đạt (x_target, 0, z_target).
    Các biến q_init = (j22, j23, j28).
    """
    def objective(q):
        # q = [j22, j23, j28]
        # Tại fk(), 6 khớp là: J20, J22, J23, J26, J28, J30
        j22, j23, j28 = q
        
        # Gọi FK, khóa cứng J20=0, J26=0, J30=0
        xt, yt, zt = fk([0.0, j22, j23, 0.0, j28, 0.0])
        
        # Tính khoảng cách (sai số). Vì J20=J26=J30=0, Y tự động bằng ~0.
        return (xt - x_target)**2 + (zt - z_target)**2
    
    # Giới hạn góc của từng khớp (theo URDF / vật lý)
    # J22: [-1.57, 1.57] 
    # J23: [-0.78, 3.91]
    # J28: [-2.35, 2.35]
    bounds = [
        (-1.57, 1.57),
        (-0.78, 3.91),
        (-2.35, 2.35)
    ]
    
    # SLSQP hỗ trợ cực tốt cho Bounds optimization
    result = minimize(objective, q_init, method='SLSQP', bounds=bounds)
    return result.x, result.fun

class PlanarTrajectoryNode(Node):
    def __init__(self):
        super().__init__('planar_trajectory_node')
        
        self.publisher = self.create_publisher(
            JointTrajectory,
            '/arm_controller/joint_trajectory',
            10
        )
        
        self.joint_names = [
            'Revolute 20', 'Revolute 22', 'Revolute 23',
            'Revolute 26', 'Revolute 28', 'Revolute 30', 'Revolute 31',
        ]
        
        self.get_logger().info("Đang tính toán quỹ đạo IK bằng Scipy...")
        self.trajectory_msg = self.compute_trajectory()
        
        # Sau 2 giây (đóng cổng kết nối pub-sub thành công), publish trajectory
        self.timer = self.create_timer(2.0, self.timer_callback)
        self.published = False

    def compute_trajectory(self):
        msg = JointTrajectory()
        msg.joint_names = self.joint_names
        
        # --- Thông số quỹ đạo ---
        # Đường thẳng đứng tại x = 0.2
        x_target = 0.2
        z_start = 0.1
        z_end = -0.1
        steps = 50
        duration_per_step = 0.1  # Mỗi điểm mất 0.1 giây
        
        q_guess = (0.0, 0.0, 0.0) # Khởi tạo IK từ vị trí Zero
        
        for i in range(steps):
            t = i / (steps - 1)
            z_curr = z_start + t * (z_end - z_start)
            
            # Giải IK bằng Scipy
            q_opt, error = ik_planar_scipy(x_target, z_curr, q_init=q_guess)
            j22, j23, j28 = q_opt
            
            # Cập nhật q_guess cho điểm tiếp theo để hàm SLSQP chạy siêu nhanh do vị trí gần nhau
            q_guess = q_opt
            
            if error > 1e-4:
                self.get_logger().warn(f"Cảnh báo: IK khó hội tụ ở z={z_curr:.3f}, err={error:.6f}m")
            
            # Khởi tạo điểm mới
            point = JointTrajectoryPoint()
            # 7 khớp: [J20, J22, J23, J26, J28, J30, J31]
            point.positions = [0.0, float(j22), float(j23), 0.0, float(j28), 0.0, 0.0]
            
            # Tính thời gian (time_from_start)
            total_time = (i + 1) * duration_per_step
            sec = int(total_time)
            nanosec = int((total_time - sec) * 1e9)
            point.time_from_start = Duration(sec=sec, nanosec=nanosec)
            
            msg.points.append(point)
            
        self.get_logger().info(f"Hoàn tất nội suy {steps} điểm IK. Tổng thời gian chạy: {steps * duration_per_step:.1f} giây.")
        return msg

    def timer_callback(self):
        if not self.published:
            self.publisher.publish(self.trajectory_msg)
            self.get_logger().info("Đã gửi JointTrajectory lệnh! Robot bắt đầu di chuyển.")
            self.published = True
            
            # Dừng script tự động sau khi xong (tổng 5s chạy IK + 2s dự phòng)
            self.get_logger().info("Chờ 7 giây để xem quỹ đạo, sau đó script sẽ tự thoát...")
            self.shutdown_timer = self.create_timer(7.0, self.shutdown_callback)

    def shutdown_callback(self):
        rclpy.shutdown()
        sys.exit(0)

def main():
    rclpy.init()
    node = PlanarTrajectoryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except SystemExit:
        pass

if __name__ == '__main__':
    main()
