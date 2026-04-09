#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64

class BatteryBalancer(Node):
    def __init__(self):
        super().__init__('battery_balancer_node')
        # Lắng nghe bản tin Float64 chứa thông số Trục X Tương lai do Dashboard nhả ra
        self.sub = self.create_subscription(Float64, '/ai_pred_com_x', self.com_cb, 10)
        self.get_logger().info('✅ Hệ thống Cân bằng Pin Tỷ lệ 1:1 đã kết nối! Đang Lắng nghe...')
        
        # Lưu lại vị trí trọng tâm của tay lúc khởi động (Trạng thái Zero)
        self.arm_x_zero = None

    def com_cb(self, msg):
        # Trọng tâm tay hiện tại / tương lai dự đoán
        arm_com_x_mm = msg.data * 1000.0
        
        # Bắt đầu khóa vị trí Zero ở lần nhận message đầu tiên
        if self.arm_x_zero is None:
            self.arm_x_zero = arm_com_x_mm
            self.get_logger().info(f"🎯 Đã khóa vị trí gốc (ZERO) của Tay: {self.arm_x_zero:+.1f} mm")
            self.get_logger().info(f"   => Pin đã được bạn setup đối xứng ở: {-self.arm_x_zero:+.1f} mm")
            return
        
        # 1. VỊ TRÍ TUYỆT ĐỐI CỦA PIN:
        # Vì M_battery (0.518kg) == M_arm (0.518kg), phương trình đòn bẩy bảo toàn Mô-men quay là:
        # X_battery = - X_arm
        abs_battery_x_mm = - arm_com_x_mm
        
        # 2. ĐỘ DỜI ĐỘNG CƠ (STEPPER TRAVEL):
        # Khi mới cắm điện, động cơ lấy vị trí ban đầu làm mốc (0).
        # Khoảng cách cần trượt = Vị trí Tuyệt đối Mới - Vị trí Tuyệt đối Ban đầu
        delta_travel_mm = abs_battery_x_mm - (-self.arm_x_zero)
        
        # Để in ra Dashboard Terminal cho dễ nhìn
        self.get_logger().info(
            f"Tay đang lao tới: [{arm_com_x_mm:+.1f} mm] "
            f"| Pin dời Tuyệt đối: [{abs_battery_x_mm:+.1f} mm] "
            f"| Motor đẩy hành trình: [{delta_travel_mm:+.1f} mm]"
        )

def main():
    rclpy.init()
    node = BatteryBalancer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
