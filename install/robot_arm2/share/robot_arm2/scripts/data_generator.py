#!/usr/bin/env python3
"""
Data Generation Script for Cartesian Mapping of 6-DOF Robot Arm.
Features:
- Analytical Forward Kinematics parser using pure XML URDF.
- Extremely fast kinematic tree traversal (No ROS TF overhead).
- Computes EE Transform and Global CoM.
- Generates data over a Cartesian grid of J22, J23, J28 holding J20, J26, J31=0.
- Append to CSV line by line.
"""
import xml.etree.ElementTree as ET
import numpy as np
import csv
import sys
import os
import itertools
from scipy.spatial.transform import Rotation as ScipyRot

def parse_xyz(s):
    return np.array(list(map(float, s.split()))) if s else np.zeros(3)

def parse_rpy(s):
    if not s: return np.eye(3)
    # URDF extrinsic XYZ matches scipy 'xyz'
    return ScipyRot.from_euler('xyz', list(map(float, s.split()))).as_matrix()

def make_transform(xyz, rot_matrix):
    T = np.eye(4)
    T[:3, :3] = rot_matrix
    T[:3, 3] = xyz
    return T

class RobotTree:
    def __init__(self, urdf_path):
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        
        self.links = {}
        self.joints = []
        self.child_to_parent_joint = {}
        self.total_mass = 0.0
        
        for link in root.findall('link'):
            name = link.get('name')
            mass = 0.0
            local_com = np.array([0., 0., 0., 1.0])
            inertial = link.find('inertial')
            if inertial is not None:
                m_el = inertial.find('mass')
                if m_el is not None:
                    mass = float(m_el.get('value', 0))
                o_el = inertial.find('origin')
                if o_el is not None:
                    local_com[:3] = parse_xyz(o_el.get('xyz', '0 0 0'))
            
            self.links[name] = {
                'mass': mass,
                'local_com': local_com,
                'T_global': np.eye(4)
            }
            self.total_mass += mass
            
        for joint in root.findall('joint'):
            jname = joint.get('name')
            jtype = joint.get('type')
            parent = joint.find('parent').get('link')
            child = joint.find('child').get('link')
            origin = joint.find('origin')
            xyz = parse_xyz(origin.get('xyz', '0 0 0')) if origin is not None else np.zeros(3)
            rpy = parse_rpy(origin.get('rpy', '0 0 0')) if origin is not None else np.eye(3)
            T_offset = make_transform(xyz, rpy)
            
            axis_el = joint.find('axis')
            axis = parse_xyz(axis_el.get('xyz', '1 0 0')) if axis_el is not None else np.array([1, 0, 0])
            
            j_data = {
                'name': jname,
                'type': jtype,
                'parent': parent,
                'child': child,
                'T_offset': T_offset,
                'axis': axis
            }
            self.joints.append(j_data)
            self.child_to_parent_joint[child] = j_data
            
        print(f"URDF parsed: {len(self.links)} links, Total Mass = {self.total_mass:.4f} kg")

        # Map children for fast tree traversal later
        self.p2c = {lnk: [] for lnk in self.links}
        for j in self.joints:
            self.p2c[j['parent']].append(j)

    def fk_all(self, q_dict):
        """Tính T_global cho tất cả các links từ base_link mượt mà qua các khớp."""
        root_link = 'base_link'
        self.links[root_link]['T_global'] = np.eye(4)
        
        # Traverse tree via BFS
        queue = [root_link]
        while queue:
            curr = queue.pop(0)
            T_parent = self.links[curr]['T_global']
            
            for j in self.p2c.get(curr, []):
                child = j['child']
                T_off = j['T_offset']
                
                # Biến đổi góc quay
                T_joint = np.eye(4)
                if j['type'] in ['revolute', 'continuous']:
                    angle = q_dict.get(j['name'], 0.0)
                    if angle != 0.0:
                        axis = j['axis']
                        rot = ScipyRot.from_rotvec(axis * angle).as_matrix()
                        T_joint[:3, :3] = rot
                
                self.links[child]['T_global'] = T_parent @ T_off @ T_joint
                queue.append(child)

    def get_com(self):
        weighted = np.zeros(3)
        for name, data in self.links.items():
            if data['mass'] > 0:
                pos_global = data['T_global'] @ data['local_com']
                weighted += data['mass'] * pos_global[:3]
        return weighted / self.total_mass
        
        
def main():
    urdf_path = '/tmp/new_arm_expanded.urdf'
    if not os.path.exists(urdf_path):
        print("Lỗi: Không tìm thấy /tmp/new_arm_expanded.urdf. Đang tự động lưu file từ xacro...")
        os.system(f"xacro {os.path.dirname(os.path.abspath(__file__))}/../urdf/new_arm/new_arm.xacro > {urdf_path}")
        
    robot = RobotTree(urdf_path)
    
    # ----------------------------------------------------
    # CHỈ ĐỊNH THÔNG SỐ GRID DATA TẠI ĐÂY
    # ----------------------------------------------------
    N = 50   # Số bước nội suy Grid
    
    low_22, high_22 = -1.570796, 1.570796
    low_23, high_23 = -0.785398, 3.909538
    low_28, high_28 = -2.356194, 2.356194
    
    j22_vals = np.linspace(low_22, high_22, N)
    j23_vals = np.linspace(low_23, high_23, N)
    j28_vals = np.linspace(low_28, high_28, N)
    
    csv_file = '/home/quan/robot_arm_uav/ros2_ws/com_dataset.csv'
    total_iters = N * N * N
    
    print(f"Bắt đầu thu thập dữ liệu Cartesian (N={N})...")
    print(f"Tổng cộng {total_iters} vector kết quả kết hợp!")
    print(f"Ghi dữ liệu vào {csv_file}")
    
    write_header = not os.path.exists(csv_file)
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                'j22', 'j23', 'j28', 
                'ee_x', 'ee_y', 'ee_z',
                'com_x', 'com_y', 'com_z',
                'R11','R12','R13',
                'R21','R22','R23',
                'R31','R32','R33'
            ])
            
        counter = 0
        # Iterator Cartesian (tất cả các tổ hợp chéo của 3 góc)
        for j22, j23, j28 in itertools.product(j22_vals, j23_vals, j28_vals):
            # Khóa cứng J20, J26, J30, J31 bằng 0 
            # để đảm bảo 100% End-Effector và CoM nằm trên mặt phẳng XZ (Y ~ 0)
            q_dict = {
                'Revolute 20': 0.0,
                'Revolute 22': j22,
                'Revolute 23': j23,
                'Revolute 26': 0.0,
                'Revolute 28': j28,
                'Revolute 30': 0.0,
                'Revolute 31': 0.0
            }
            
            # 1. Tính toán cây FK offline
            robot.fk_all(q_dict)
            
            # 2. Lấy Tọa độ và Ma trận quay End-effector (từ ngòi bút: but_1)
            T_ee = robot.links['but_1']['T_global']
            ee_x, ee_y, ee_z = T_ee[:3, 3]
            R = T_ee[:3, :3].flatten()
            
            # 3. Tính toán trọng tâm toàn phần CoM
            com = robot.get_com()
            
            # 4. Lưu lại
            writer.writerow([
                f"{j22:.5f}", f"{j23:.5f}", f"{j28:.5f}",
                f"{ee_x:.5f}", f"{ee_y:.5f}", f"{ee_z:.5f}",
                f"{com[0]:.5f}", f"{com[1]:.5f}", f"{com[2]:.5f}",
                *map(lambda x: f"{x:.5f}", R)
            ])
            
            counter += 1
            if counter % 10000 == 0:
                print(f"  + Đã tính xong {counter}/{total_iters} ({(counter/total_iters)*100:.1f}%)")
                
    print("✓ Hoàn tất xử lý data đợt này!")

if __name__ == '__main__':
    main()
