#!/usr/bin/env python3
"""
Custom Analytical Forward Kinematics parser using pure XML URDF.
Recovered module containing RobotTree for FK and CoM calculations.
"""
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation as ScipyRot

def parse_xyz(s):
    return np.array(list(map(float, s.split()))) if s else np.zeros(3)

def parse_rpy(s):
    if not s: return np.eye(3)
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

        self.p2c = {lnk: [] for lnk in self.links}
        for j in self.joints:
            self.p2c[j['parent']].append(j)

    def fk_all(self, q_dict):
        root_link = 'base_link'
        self.links[root_link]['T_global'] = np.eye(4)
        
        queue = [root_link]
        while queue:
            curr = queue.pop(0)
            T_parent = self.links[curr]['T_global']
            
            for j in self.p2c.get(curr, []):
                child = j['child']
                T_off = j['T_offset']
                
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
