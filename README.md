# UAV Robot Arm 3D Center of Mass (CoM) Prediction using 1D-CNN

This project applies a deep learning **1-Dimensional Convolutional Neural Network (1D-CNN)** built on **ROS 2 Humble** and **Ignition Gazebo Fortress** to predict the future spatial shift of the Center of Mass (CoM) on a 6-DOF (Degrees of Freedom) robot arm. 

### Key Features (V5 3D Spatial Update):
- Full 3D Spatial prediction for the Arm's CoM.
- Pristine 1D-CNN single-loop training architecture achieving an impressive Validation Loss of **0.0285**.
- Integrated, highly intuitive Real-time Dashboard featuring **7 Simultaneous Plots**.

---

## 🛠 Latest System Architecture (`armfinal_description`)

Current Digital Twin: **`armfinal_description`**

In this repository, the robot arm uses a highly accurate CAD structure:
- **Geometry:** Native `.stl` visual and collision meshes.
- **Inertia & Mass:** Perfectly measured mass and center of mass parameters for every individual link, ensuring hyper-stable dynamics at extreme rotational speeds without jitter or self-destruction.
- **Controller:** Operates via `gz_ros2_control` with aggressively tuned PID profiles (P-Gain 150) and micro-damping (0.01) to completely eliminate tracking latency.

---

## 🚀 Execution Guide

### 1. Launch Gazebo Simulation
The Gazebo environment must be instantiated first to activate the `ros2_control` hardware interface:
```bash
cd ~/robot_arm_uav/ros2_ws
colcon build --packages-select armfinal_description
source install/setup.bash
ros2 launch armfinal_description armfinal_gazebo.launch.py
```

### 2. Launch AI Real-time Dashboard
This dashboard listens to the `/joint_states` topic and feeds the data into the trained `best_com_model_armfinal.pth` model (trained on 6,400 samples) to visualize two synchronized pipelines:
- 🔴 **Ground Truth CoM:** The current physical center of mass.
- 🔵 **AI Predicted CoM:** The network's prediction of the CoM exactly 0.5 seconds into the future.

*(Run in Terminal 2)*
```bash
source ~/robot_arm_uav/ros2_ws/install/setup.bash
python3 ~/robot_arm_uav/ros2_ws/src/armfinal_description/scripts/com_dashboard_armfinal.py
```

### 3. Manual Joint Slider Panel
Use this slider to inject high-difficulty mechanical trajectories into the robot.
As you drag the sliders, observe Terminal 2 to see how the **Blue Dashed Line (AI)** dynamically leads the **Red Solid Line (Truth)** by precisely 0.5 seconds!

*(Run in Terminal 3)*
```bash
source ~/robot_arm_uav/ros2_ws/install/setup.bash
python3 ~/robot_arm_uav/ros2_ws/src/armfinal_description/scripts/armfinal_joint_slider.py
```

### 4. Offline Test Set Evaluation
This command extracts the first 300 samples of the Test Set (data entirely hidden from the CNN during training) to cross-validate and measure RMSE & R² metrics across the X, Y, and Z axes. The statistical results will be printed to the terminal alongside high-resolution Matplotlib scatter figures.
*(Run in Terminal 4)*
```bash
source ~/robot_arm_uav/ros2_ws/install/setup.bash
python3 ~/robot_arm_uav/ros2_ws/src/armfinal_description/scripts/evaluate_armfinal.py
```

### 5. 3D Workspace Topology Simulator
This script triggers a Monte-Carlo uniform sampler (50,000 nodes) routed through the mathematical Forward Kinematics (FK) engine to project the complete unreachable "Wall Space" boundary, encased by 2D/3D Convex Hulls. This visualization mathematically proves the arm's trajectory envelope coverage.
*(Run in Terminal 5)*
```bash
source ~/robot_arm_uav/ros2_ws/install/setup.bash
python3 ~/robot_arm_uav/ros2_ws/src/armfinal_description/scripts/workspace_simulator.py
```

### 6. Hardware Battery Balancer Actuation Bridge
This command boots a standalone ROS 2 Subscriber listening exclusively to the Absolute Future X prediction from the Dashboard. It dynamically solves the 1:1 mass lever-arm moment equation and pumps out Delta Displacement commands (mm). This serves as the physical pipeline wired directly to the Linear Actuator motor controlling the UAV's counter-weight battery.
*(Run in Terminal 6 - Recommended parallel with Terminal 2)*
```bash
source ~/robot_arm_uav/ros2_ws/install/setup.bash
python3 ~/robot_arm_uav/ros2_ws/src/armfinal_description/scripts/battery_balancer_listener.py
```

---

## 🧬 Overview: Dynamic Data Collection Pipeline

The secret behind the AI's success lies in the **Robot Dynamic Dance** process within Gazebo. Instead of manually recording rigid waypoints, the system auto-generates high-fidelity physics-bound data via the following flow:

1. **Sinusoidal Trajectory Generator:**
   The `dynamic_trajectory.py` node continuously bombards the Controller with Position commands driven by superimposed Sin/Cos waves featuring randomized Frequencies and Amplitudes spanning $0.1$ to $1.2\text{Hz}$. This creates a chaotic 3D dance covering every quadrant of the Workspace Topology. The Robot dances for **10 routines**, lasting exactly **300 seconds (5 minutes)**.

2. **Spatio-Temporal Synchronization:**
   The `dynamic_data_recorder.py` background daemon captures absolute snapshots at **20 Frames-per-second (20Hz)**.  
   At every millisecond, the recorder rips **9 core features** out of the physical joints (Joint 2, 4, 5 Positions; Joint Velocities; and End-Effector XYZ). Relying on the `RobotTree` Forward Kinematics solver, it tags the absolute real-time Ground Truth **CoM X, Y, Z** mapping directly to the future horizon.

3. **Static Boundary Condition Anchoring:**
   Within the CSV dataset, a golden ratio of `~6%` Idle Samples encapsulates the start and end periods. This guarantees the AI network inherently memorizes the **Zero-Velocity Inertia Law**: *"If velocity = 0, the future t+0.5 position strictly equals the current position"*, preventing UI drift when human joysticks are released. This yields the ultra-clean 6,400-row `armfinal_dynamic_com_dataset.csv`.

---

## 🧠 Automating Retraining From Scratch

The entire AI Pipeline is natively automated. If you modify joint lengths or mass links in the future, follow this sequence:
1. Launch Gazebo (Step 1 above).
2. Execute the data collection dance (5-minute routine):
   `python3 src/armfinal_description/scripts/dynamic_trajectory.py`
3. Concurrently launch the CSV data recorder daemon:
   `python3 src/armfinal_description/scripts/dynamic_data_recorder.py`
4. Preprocessing (Generate target tensors via Sliding Windows):
   `python3 src/armfinal_description/scripts/preprocess_armfinal.py`
5. Retrain the PyTorch PyTorch Network:
   `python3 src/armfinal_description/scripts/train_armfinal.py`

---

## 📊 System Architecture Flow Diagrams

Below are the 4 fundamental lifecycles of the project (pre-rendered into offline `.png` images within the `/images` directory):

### 1. Data Generation Pipeline
![Data Pipeline](file:///home/quan/robot_arm_uav/images/data_pipeline.png)

### 2. ML Training Pipeline
![Training Pipeline](file:///home/quan/robot_arm_uav/images/training_pipeline.png)

### 3. Real-time Inference Pipeline
![Inference Pipeline](file:///home/quan/robot_arm_uav/images/inference_pipeline.png)

### 4. Hardware Battery Balancer Logic
![Hardware Balancer](file:///home/quan/robot_arm_uav/images/hardware_balancer.png)
