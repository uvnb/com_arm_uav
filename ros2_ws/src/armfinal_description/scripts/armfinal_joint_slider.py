#!/usr/bin/env python3
"""
Slider-based GUI to control the 6 joints of armfinal in Gazebo
via the arm_controller (JointTrajectoryController).
"""
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

# Import tkinter AFTER rclpy.init() to avoid potential issues
import tkinter as tk

class JointSliderGUI(Node):
    def __init__(self):
        super().__init__('armfinal_joint_slider_gui')

        # Arm joints and their limits (from armfinal.ros2_control.xacro)
        # Order matches armfinal_controllers.yaml
        self.joints = {
            'Revolute 2':  {'min': -3.14, 'max': 3.14, 'label': 'Joint 1 (Base)'},
            'Revolute 4':  {'min': -1.57, 'max': 1.57, 'label': 'Joint 2 (Shoulder)'},
            'Revolute 5':  {'min': -1.57, 'max': 1.57, 'label': 'Joint 3 (Elbow)'},
            'Revolute 8':  {'min': -1.57, 'max': 1.57, 'label': 'Joint 4 (Wrist 1)'},
            'Revolute 10': {'min': -1.57, 'max': 1.57, 'label': 'Joint 5 (Wrist 2)'},
            'Revolute 12': {'min': -1.57, 'max': 1.57, 'label': 'Joint 6 (Wrist 3)'},
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
        self.root.title('ArmFinal Joint Controller')
        self.root.configure(bg='#1e1e1e')
        self.root.geometry('550x550')

        title = tk.Label(
            self.root, text='ArmFinal Manual Control',
            font=('Segoe UI', 16, 'bold'), fg='#00ff99', bg='#1e1e1e'
        )
        title.pack(pady=20)

        for joint_name, info in self.joints.items():
            frame = tk.Frame(self.root, bg='#1e1e1e')
            frame.pack(fill='x', padx=30, pady=5)

            label = tk.Label(
                frame, text=info['label'],
                font=('Segoe UI', 10), fg='#cccccc', bg='#1e1e1e',
                width=20, anchor='w'
            )
            label.pack(side='left')

            slider = tk.Scale(
                frame,
                from_=info['min'], to=info['max'],
                resolution=0.01, orient='horizontal',
                length=300,
                bg='#333333', fg='#ffffff',
                troughcolor='#00ff99', highlightthickness=0,
                command=lambda val, jn=joint_name: self.on_slider_change(jn, float(val))
            )
            slider.set(0.0)
            slider.pack(side='right')
            self.sliders[joint_name] = slider

        # Button frame
        btn_frame = tk.Frame(self.root, bg='#1e1e1e')
        btn_frame.pack(pady=30)

        center_btn = tk.Button(
            btn_frame, text='RESET TO ZERO', font=('Segoe UI', 11, 'bold'),
            bg='#00ff99', fg='#1e1e1e', padx=30, pady=10,
            activebackground='#00cc7a', relief='flat',
            command=self.center_all
        )
        center_btn.pack()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(100, self.ros_spin_once)

    def ros_spin_once(self):
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
        point.time_from_start = Duration(sec=0, nanosec=50000000)  # 0.05s to ensure smooth continuous slider dragging

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
