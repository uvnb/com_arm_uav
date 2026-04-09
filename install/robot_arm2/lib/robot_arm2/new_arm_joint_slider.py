#!/usr/bin/env python3
"""
Simple slider-based GUI to control the 7 arm joints of new_arm in Gazebo
via the arm_controller (JointTrajectoryController).

Uses timer-based ROS spinning to avoid segfault from threading conflicts.
"""
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

# Import tkinter AFTER rclpy.init() to avoid segfault
import tkinter as tk


class JointSliderGUI(Node):
    def __init__(self):
        super().__init__('joint_slider_gui')

        # Arm joints and their limits (from xacro)
        self.joints = {
            'Revolute 20': {'min': -3.14, 'max': 3.14, 'label': 'J1 Base Rotation'},
            'Revolute 22': {'min': -1.57, 'max': 1.57, 'label': 'J2 Shoulder'},
            'Revolute 23': {'min': -0.79, 'max': 3.91, 'label': 'J3 Elbow'},
            'Revolute 26': {'min': -3.14, 'max': 3.14, 'label': 'J4 Wrist 1'},
            'Revolute 28': {'min': -2.36, 'max': 2.36, 'label': 'J5 Wrist 2'},
            'Revolute 30': {'min': 0.0,   'max': 3.14, 'label': 'J6 Gripper Tilt'},
            'Revolute 31': {'min': -3.14, 'max': 3.14, 'label': 'J7 End Effector'},
        }

        self.publisher = self.create_publisher(
            JointTrajectory,
            '/arm_controller/joint_trajectory',
            10
        )

        self.slider_values = {j: 0.0 for j in self.joints}
        self.sliders = {}
        self.setup_gui()

    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title('New Arm Joint Controller')
        self.root.configure(bg='#2b2b2b')
        self.root.geometry('520x580')

        title = tk.Label(
            self.root, text='New Arm Joint Controller',
            font=('Arial', 14, 'bold'), fg='#00ccff', bg='#2b2b2b'
        )
        title.pack(pady=10)

        for joint_name, info in self.joints.items():
            frame = tk.Frame(self.root, bg='#2b2b2b')
            frame.pack(fill='x', padx=20, pady=3)

            label = tk.Label(
                frame, text=info['label'],
                font=('Arial', 10), fg='#ffffff', bg='#2b2b2b',
                width=18, anchor='w'
            )
            label.pack(side='left')

            slider = tk.Scale(
                frame,
                from_=info['min'], to=info['max'],
                resolution=0.01, orient='horizontal',
                length=280,
                bg='#3c3c3c', fg='#ffffff',
                troughcolor='#555555', highlightthickness=0,
                command=lambda val, jn=joint_name: self.on_slider_change(jn, float(val))
            )
            slider.set(0.0)
            slider.pack(side='right')
            self.sliders[joint_name] = slider

        # Center all button
        btn_frame = tk.Frame(self.root, bg='#2b2b2b')
        btn_frame.pack(pady=15)

        center_btn = tk.Button(
            btn_frame, text='Center All', font=('Arial', 11, 'bold'),
            bg='#00ccff', fg='#000000', padx=20, pady=5,
            command=self.center_all
        )
        center_btn.pack()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Use tkinter's after() to spin ROS 2 in the same thread
        self.root.after(100, self.ros_spin_once)

    def ros_spin_once(self):
        """Spin ROS 2 from tkinter's main loop to avoid threading issues."""
        rclpy.spin_once(self, timeout_sec=0.01)
        self.root.after(50, self.ros_spin_once)

    def on_slider_change(self, joint_name, value):
        self.slider_values[joint_name] = value
        self.publish_trajectory()

    def publish_trajectory(self):
        msg = JointTrajectory()
        msg.joint_names = list(self.joints.keys())

        point = JointTrajectoryPoint()
        point.positions = [self.slider_values[j] for j in self.joints]
        point.time_from_start = Duration(sec=0, nanosec=500000000)  # 0.5s

        msg.points = [point]
        self.publisher.publish(msg)

    def center_all(self):
        for joint_name, slider in self.sliders.items():
            slider.set(0.0)
            self.slider_values[joint_name] = 0.0
        self.publish_trajectory()

    def on_close(self):
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main():
    rclpy.init()
    gui = JointSliderGUI()
    try:
        gui.run()
    except KeyboardInterrupt:
        pass
    finally:
        gui.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
