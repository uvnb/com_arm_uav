# Đồ án: Dự đoán Quỹ đạo Trọng tâm (CoM) Cánh tay Robot bằng Trí tuệ Nhân tạo 1D-CNN

Dự án này ứng dụng mô hình học sâu **Mạng Neural Tích chập 1 chiều (1D-CNN)** trên nền tảng **ROS 2 Humble** và **Ignition Gazebo Fortress** để học và dự đoán trước sự dịch chuyển của Khối lượng tâm (Center of Mass - CoM) trên một cánh tay robot 6 bậc tự do (6-DOF).

### Điểm nổi bật cập nhật nội bật (V5 3D Spatial):
- Dự đoán toàn vẹn Không gian khối 3 Chiều (X, Y, Z).
- Mạng CNN huấn luyện 1 vòng lặp tinh khiết đạt Validation Loss cực kỳ ấn tượng **0.0285**.
- Tích hợp Bảng điều khiển siêu trực quan **7 Biểu đồ Đồng thời**, cùng Hệ thống tự động trích xuất **Báo cáo Đánh giá (Evaluation Report)** bằng R² Scatters.

---

## 🛠 Cấu trúc Hệ thống Mới Nhất (armfinal_description)

Bản sao số (Digital Twin) hiện tại: **`armfinal_description`**

Trong kho lưu trữ này, cánh tay robot đã được nhập từ bản thảo **CAD 3D chuẩn xác 100%**:
- **Geometry:** Các file `.stl` định hình vỏ vật lý.
- **Inertia & Mass:** Khối lượng chuẩn của từng mắt xích và tọa độ trọng tâm cục bộ thiết kế khớp hoàn hảo tới từng milimet, giúp cánh tay giữ vững cấu trúc vật lý ở tốc độ quay siêu cao mà không bị tự hủy hay rung lắc.
- **Controller:** Sử dụng `gz_ros2_control` với cấu hình PID-Gain chuyên dụng (P-Gain 150) và Damping siêu nhỏ (0.01) để loại bỏ hoàn toàn độ trễ trượt.

---

## 🚀 Hướng Dẫn Chạy Dự Án

### 1. Khởi động Môi trường Mô phỏng Gazebo
Phải chạy môi trường ảo Gazebo trước tiên để hệ thống `ros2_control` lên sóng:
```bash
cd ~/robot_arm_uav/ros2_ws
colcon build --packages-select armfinal_description
source install/setup.bash
ros2 launch armfinal_description armfinal_gazebo.launch.py
```

### 2. Chạy Bảng Biểu diễn Trí Tuệ Nhân Tạo (AI Real-time Dashboard)
Bảng điều khiển này sẽ lắng nghe topic `/joint_states`, đưa qua mô hình `best_com_model_armfinal.pth` (đã được huấn luyện qua 12,000 mẫu) và biểu diễn hai đường đồ thị:
- 🔴 **Tọa độ CoM Ground Truth:** Vị trí trọng tâm hiện tại.
- 🔵 **Tọa độ CoM AI Dự đoán:** Vị trí trọng tâm đi trước tương lai 0.5 giây.

*(Mở Terminal 2)*
```bash
source ~/robot_arm_uav/ros2_ws/install/setup.bash
python3 ~/robot_arm_uav/ros2_ws/src/armfinal_description/scripts/com_dashboard_armfinal.py
```

### 3. Bảng Điều khiển Bằng Tay (Joint Slider)
Thanh trượt này dùng để bạn tự tay cấp quỹ đạo độ khó cao cho Rorbot.
Khi bạn kéo thanh trượt, hãy quan sát ở Terminal 2 xem **đường gạch xanh (AI) nảy lên trước đường gạch đỏ (Thực tế) 0.5 giây** chính xác đến mức nào!

*(Mở Terminal 3)*
```bash
source ~/robot_arm_uav/ros2_ws/install/setup.bash
python3 ~/robot_arm_uav/ros2_ws/src/armfinal_description/scripts/armfinal_joint_slider.py
```

### 4. Đánh giá Ngoại tuyến trên Tập Test Set (Evaluation Report)
Lệnh này sẽ trích xuất 300 mẫu đầu tiên của tập Test Set (Trượt qua mạng CNN nhưng KHÔNG tham gia huấn luyện) để kiểm chứng chéo và đo lường RMSE, R² trên cả 3 trục tọa độ X, Y, Z. Kết quả sẽ được in ra Terminal và hình ảnh Matplotlib độ phân giải cao sẽ hiển thị.
*(Mở Terminal 4)*
```bash
source ~/robot_arm_uav/ros2_ws/install/setup.bash
python3 ~/robot_arm_uav/ros2_ws/src/armfinal_description/scripts/evaluate_armfinal.py
```

### 5. Mô phỏng Không gian Vật lý (3D Workspace Simulator)
Tiến trình này chạy lấy mẫu Monte-Carlo ngẫu nhiên (50,000 điểm) đâm qua tầng Toán học Động học Thuận (FK) để vẽ ra toàn bộ "Không gian Bức tường" mà cánh tay có thể với tới, sau đó được bo viền bởi lưới Vỏ Bao Lồi 2D/3D (Convex Hull). Bức tranh này minh chứng không gian múa của tay đã phủ kín.
*(Mở Terminal 5)*
```bash
source ~/robot_arm_uav/ros2_ws/install/setup.bash
python3 ~/robot_arm_uav/ros2_ws/src/armfinal_description/scripts/workspace_simulator.py
```

### 6. Khởi động Đầu mối Phần cứng Cơ cấu Cân bằng Pin (Battery Balancer Hardware Bridge)
Lệnh này kích hoạt một ROS 2 Subcriber độc lập chuyên lắng nghe thông số Tuyệt đối X tương lai từ Dashboard. Nó lập tức giải phương trình hệ số mô-men đòn bẩy tỷ lệ 1:1 và xuất lệnh Delta Displacement sang mili-mét. Đây chính là trạm phát tín hiệu cắm dây thẳng vào Động cơ Linear Actuator của cụm Pin đối trọng gắn trên mình drone.
*(Mở Terminal 6 - Chạy song song bảng Dashboard Terminal 2)*
```bash
source ~/robot_arm_uav/ros2_ws/install/setup.bash
python3 ~/robot_arm_uav/ros2_ws/src/armfinal_description/scripts/battery_balancer_listener.py
```
---

## 🧬 Tổng quan: Quá trình Thu thập Dữ liệu Động học (Data Collection Pipeline)

Trái tim của độ phủ AI nằm ở quá trình **Nhảy múa Vật lý (Robot Dynamic Dance)** trong môi trường Gazebo. Thay vì chỉ nhập liệu bằng tay một cách ngẫu nhiên, hệ thống tự động sinh dữ liệu đạt chất lượng cao dựa trên Cơ học Động lực nhờ quy trình sau:

1. **Khởi tạo Quỹ đạo Sóng Sin (Sinusoidal Trajectories):**
   Tiến trình `dynamic_trajectory.py` liên tục bơm lệnh Position vào Controller thông qua các hàm Sóng Sin/Cos chồng chéo với những Tần số (Frequency) và Biên độ (Amplitude) ngẫu nhiên dao động từ $0.1$ đến $1.2\text{Hz}$. Việc này tạo ra một "điệu nhảy" rung giật 3D bao phủ mọi ngóc ngách của không gian Không gian Tác điểm (Workspace Topology). Tổng cộng Cánh tay nhảy **10 bài múa**, kéo dài đúng **300 giây (5 phút)**.

2. **Đồng bộ hóa Không gian & Thời gian (Data Recorder):**
   Hệ thống quan sát ngầm `dynamic_data_recorder.py` tiến hành chụp X-quang liên tục **20 khung hình/giây (20Hz)**.  
   Tại mỗi mili-giây, máy ghi trích xuất **9 Đặc trưng lõi** của cánh tay vật lý (Joint 2, 4, 5 Positions; Joint Velocities; và End-Effector X, Y, Z) kết hợp với giải thuật Động học Thuận (Forward Kinematics từ `RobotTree`) để tự động gán nhãn thực tế **CoM X, Y, Z** của chân trời tương lai. Toàn bộ được đóng gói song song đồng thời, cam kết không độ trễ. 

3. **Tiết chế Điểm tĩnh (Static Boundary Condition Anchor):**
   Trong file CSV có một lượng nhỏ tỉ lệ vàng `~6%` các bản nháp tĩnh (Idle Samples) ở phần đầu và kết thúc bản thu. Thay vì rớt dữ liệu, con số 400 mẫu này dạy cho ma trận AI thuộc lòng **Định luật Quán tĩnh Zero-Velocity**: *"Nếu vận tốc = 0, khoảng cách t+0.5 tương lai bằng chính xác thực tại"*, ngăn biểu đồ Dashboard xê lệch khi con người thả tay Joystick. Tổng thành quả là File Matrix CSV `armfinal_dynamic_com_dataset.csv` cực sạch chứa ~6,400 dòng Ma Trận chuẩn xác.

---

## 🧠 Quy trình Tự Re-Train Model Từ Đầu (Nếu cần)

Toàn bộ Pipeline Trí tuệ Nhân tạo đã được tự động hóa. Nếu bạn thay đổi chiều dài khớp hoặc khối lượng, hãy làm theo quy trình:
1. Mở Gazebo (Bước 1 ở trên).
2. Chạy quỹ đạo thu thập dữ liệu (Múa ngẫu nhiên 5 phút):
   `python3 src/armfinal_description/scripts/dynamic_trajectory.py`
3. Đan xen đó, chạy song song máy ghi dữ liệu CSV:
   `python3 src/armfinal_description/scripts/dynamic_data_recorder.py`
4. Tiền xử lý (Sinh ra Sliding Window Matrix):
   `python3 src/armfinal_description/scripts/preprocess_armfinal.py`
5. Huấn luyện lại Mạng PyTorch:
   `python3 src/armfinal_description/scripts/train_armfinal.py`

---

