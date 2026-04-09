#!/usr/bin/env python3
"""
Global Center of Mass Calculator for ROS 2.

Usage:
  python3 calculate_global_com.py 2>/dev/null

The '2>/dev/null' suppresses TF_OLD_DATA C++ warnings from stderr.
CoM data is written to 'robot_com_log.csv' and printed to stdout.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TransformStamped
import xml.etree.ElementTree as ET
import numpy as np
import sys
import csv
import os
from rclpy.qos import QoSProfile, DurabilityPolicy


def transform_to_matrix(tf_msg: TransformStamped) -> np.ndarray:
    t = tf_msg.transform.translation
    q = tf_msg.transform.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    R = np.array([
        [1-2*y**2-2*z**2,   2*x*y-2*w*z,   2*x*z+2*w*y],
        [  2*x*y+2*w*z, 1-2*x**2-2*z**2,   2*y*z-2*w*x],
        [  2*x*z-2*w*y,   2*y*z+2*w*x, 1-2*x**2-2*y**2],
    ])
    M = np.eye(4)
    M[:3, :3] = R
    M[:3,  3] = [t.x, t.y, t.z]
    return M


class GlobalCoMCalculator(Node):
    def __init__(self):
        super().__init__('global_com_calculator')

        # Không force use_sim_time - để ROS tự quyết định
        # Dùng Time(0) để luôn lấy transform mới nhất trong buffer

        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.urdf_parsed  = False
        self.links_mass   = {}           # {link_name: mass (kg)}
        self.links_local_com = {}        # {link_name: np.array([x,y,z,1])}
        self.total_mass   = 0.0
        # Thứ tự ưu tiên root frame
        # 'base_link' là root thực tế vì Ignition Gazebo không public 'world→base_link' vào TF
        self.PREFERRED_ROOTS = ['base_link', 'world', 'odom', 'map']
        self.root_frame   = None         # root TF frame được auto-detect

        # CSV output
        self.csv_path = 'robot_com_log.csv'
        with open(self.csv_path, 'w', newline='') as f:
            csv.writer(f).writerow(['timestamp_sec', 'com_x', 'com_y', 'com_z'])

        # Subscribe URDF
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, '/robot_description',
                                 self._urdf_cb, qos_profile=qos)

        # Timer 10 Hz
        self.create_timer(0.1, self._compute_cb)

        # Timer chẩn đoán 5 giây một lần để check frame
        self.create_timer(5.0, self._diagnose_cb)

        self._spinner_idx = 0
        print(f"CSV -> {os.path.abspath(self.csv_path)}", flush=True)
        print("Đang chờ URDF và TF data...", flush=True)

    # -------------------------------------------------------------------------
    def _urdf_cb(self, msg: String):
        if self.urdf_parsed:
            return
        try:
            root = ET.fromstring(msg.data)
            for link in root.findall('link'):
                name = link.get('name')
                inertial = link.find('inertial')
                if inertial is None:
                    continue
                m_el = inertial.find('mass')
                if m_el is None:
                    continue
                m = float(m_el.get('value', 0))
                if m <= 0:
                    continue
                orig = inertial.find('origin')
                xyz  = list(map(float, orig.get('xyz', '0 0 0').split())) if orig is not None else [0,0,0]
                self.links_mass[name]      = m
                self.links_local_com[name] = np.array([*xyz, 1.0])
                self.total_mass           += m

            self.urdf_parsed = True
            print(f"URDF OK: {len(self.links_mass)} links, total mass = {self.total_mass:.4f} kg", flush=True)
        except Exception as e:
            self.get_logger().error(f"URDF parse error: {e}")

    # -------------------------------------------------------------------------
    def _diagnose_cb(self):
        """Mỗi 5 giây in ra danh sách frame đang có trong TF buffer."""
        try:
            frames_str = self.tf_buffer.all_frames_as_string()
            frame_names = []
            for line in frames_str.splitlines():
                line = line.strip()
                if not line.startswith('Frame'):
                    continue
                # Format: "Frame <name> exists with parent <parent>."
                parts = line.split()
                if len(parts) >= 2:
                    frame_names.append(parts[1])

            if frame_names:
                if self.root_frame is None:
                    for candidate in self.PREFERRED_ROOTS:
                        if candidate in frame_names:
                            self.root_frame = candidate
                            print(f"\nAuto-detect root frame: '{self.root_frame}'", flush=True)
                            break
                    if self.root_frame is None:
                        self.root_frame = frame_names[0]
                        print(f"\nRoot frame (fallback): '{self.root_frame}'", flush=True)
                print(f"\n[TF] {len(frame_names)} frames, root='{self.root_frame}': {', '.join(frame_names[:8])}...", flush=True)
            else:
                print("\n[TF] Không có frame nào! Hãy đảm bảo Gazebo đang chạy và Play.", flush=True)
        except Exception as e:
            print(f"\n[TF diag error] {e}", flush=True)

    # -------------------------------------------------------------------------
    def _compute_cb(self):
        if not self.urdf_parsed or self.total_mass <= 0:
            return
        if self.root_frame is None:
            return  # Chưa biết root frame, chờ diagnose

        weighted = np.zeros(3)
        avail_mass = 0.0
        sync_time = None

        for link_name, mass in self.links_mass.items():
            try:
                if sync_time is None:
                    # Lấy transform mới nhất của link đầu tiên làm mốc thời gian chung (sync_time)
                    tf_msg = self.tf_buffer.lookup_transform(
                        self.root_frame, link_name, rclpy.time.Time())
                    sync_time = rclpy.time.Time.from_msg(tf_msg.header.stamp)
                else:
                    # Bắt buộc các link sau phải được nội suy (interpolate) đúng tại sync_time
                    tf_msg = self.tf_buffer.lookup_transform(
                        self.root_frame, link_name, sync_time)

                M   = transform_to_matrix(tf_msg)
                pos = np.dot(M, self.links_local_com[link_name])
                weighted   += mass * pos[:3]
                avail_mass += mass
            except Exception:
                # Bỏ qua nếu không tìm thấy transform tại sync_time
                pass

        if avail_mass <= 0:
            return

        com = weighted / avail_mass
        
        # Dùng sync_time để ghi timestamp cho chính xác
        if sync_time is not None:
            ts = sync_time.nanoseconds * 1e-9
        else:
            now = self.get_clock().now()
            ts  = now.nanoseconds * 1e-9

        # Ghi CSV
        with open(self.csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([f"{ts:.3f}", f"{com[0]:.6f}", f"{com[1]:.6f}", f"{com[2]:.6f}"])

        # In terminal (đè trên 1 dòng)
        SPIN = ['-', '\\', '|', '/']
        self._spinner_idx = (self._spinner_idx + 1) % 4
        pct  = int(avail_mass / self.total_mass * 100)
        line = (f"[{SPIN[self._spinner_idx]}] "
                f"Global CoM ({self.root_frame}) | "
                f"X:{com[0]:>7.4f}  Y:{com[1]:>7.4f}  Z:{com[2]:>7.4f}  "
                f"[{pct}% mass resolved]")
        sys.stdout.write('\r' + line + '   ')
        sys.stdout.flush()


# -----------------------------------------------------------------------------
def main():
    rclpy.init()
    node = GlobalCoMCalculator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
