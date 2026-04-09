"""
Forward Kinematics for the NEW 6-DOF UAV Robot Arm (new_arm.xacro)
Pure Python implementation (NO numpy) to prevent tkinter segfaults!

Kinematic chain from URDF (with structural 180-degree yaw applied at Rigid 5/6):
  base_link
    → Rigid 6 (yaw=π, xyz flipped) → que2_1
    → Rigid 18 → base_1
    → Rigid 19 → DigitalServo8120_1
    → **Revolute 20** (axis: -Z)            → link1_1
    → Rigid 21 → DigitalServo8120__1__1
    → **Revolute 22** (axis: -Y)            → link2_1
    → **Revolute 23** (axis: -Y)            → DigitalServo8120__2__1
    → Rigid 24 → link3_1
    → Rigid 25 → servoMG90S_1
    → **Revolute 26** (axis: -X)            → link4_1
    → Rigid 27 → servoMG90S__1__1
    → **Revolute 28** (axis: -Y)            → cuoi_1
    → Rigid 29 → servoMG90S__2__1
    → **Revolute 30** (axis: +Y)            → hopbut_1
    → Revolute 31 (continuous pen spin, -Z) → but_1
"""

import math
from typing import Tuple

# ============================================================================
# Joint limits (from new_arm.xacro)
# ============================================================================
JOINT_LIMITS_LOW = (
    -3.141593,   # Revolute 20
    -3.141593,   # Revolute 22
    -3.141593,   # Revolute 23
    -3.141593,   # Revolute 26
    -3.141593,   # Revolute 28
    -3.141593,   # Revolute 30
)
JOINT_LIMITS_HIGH = (
     3.141593,   # Revolute 20
     3.141593,   # Revolute 22
     3.141593,   # Revolute 23
     3.141593,   # Revolute 26
     3.141593,   # Revolute 28
     3.141593,   # Revolute 30
)

# ============================================================================
# DH-like offset tables  (xyz from URDF origin tags)
# ============================================================================

_BASE_TO_REV20 = (
    0.046528 + 0.242527 - 0.198417 + 0.034687,   # X:  0.125325
   -0.031724 + 0.123276 - 0.095138 + 0.003900,   # Y:  0.000314
   -0.001109 - 0.026109 + 0.009859 - 0.016200,   # Z: -0.033559
)

_REV20_TO_REV22 = (
    -0.048931 + 0.034687,   # X: -0.014244
    -0.007000 - 0.019200,   # Y: -0.026200
    -0.033724 - 0.003900,   # Z: -0.037624
)

_REV22_TO_REV23 = (0.0, 0.0, -0.155)

_REV23_TO_REV26 = (
    -0.034687 - 0.042816 - 0.014800,   # X: -0.092303
     0.019200 + 0.036200 - 0.009950,   # Y:  0.045450
     0.003900 - 0.033750 + 0.000000,   # Z: -0.029850
)

_REV26_TO_REV28 = (
    -0.042500 + 0.000000,   # X: -0.042500
    -0.023000 - 0.014800,   # Y: -0.037800
    -0.015200 + 0.009950,   # Z: -0.005250
)

_REV28_TO_REV30 = (
    -0.075000 + 0.000000,   # X: -0.075000
     0.007500 + 0.015000,   # Y:  0.022500
     0.015200 - 0.020450,   # Z: -0.005250
)

# Revolute 30 → End-effector tip (but_1 pen tip)
# Rev31 origin + pen tip extension = z=-0.01 - 0.049 = -0.059
_REV30_TO_EE = (0.0, 0.01225, -0.059)

# Joint rotation axes (from URDF)
JOINT_AXES = ['neg_z', 'neg_y', 'neg_y', 'neg_x', 'neg_y', 'pos_y']

_OFFSETS = [
    _BASE_TO_REV20, _REV20_TO_REV22, _REV22_TO_REV23,
    _REV23_TO_REV26, _REV26_TO_REV28, _REV28_TO_REV30, _REV30_TO_EE,
]

# ============================================================================
# Math helpers (Pure Python)
# ============================================================================

def _add(v1, v2):
    return (v1[0]+v2[0], v1[1]+v2[1], v1[2]+v2[2])

def _matmul(M, v):
    return (
        M[0][0]*v[0] + M[0][1]*v[1] + M[0][2]*v[2],
        M[1][0]*v[0] + M[1][1]*v[1] + M[1][2]*v[2],
        M[2][0]*v[0] + M[2][1]*v[1] + M[2][2]*v[2]
    )

def _mat_mul_mat(A, B):
    return (
        (A[0][0]*B[0][0]+A[0][1]*B[1][0]+A[0][2]*B[2][0], A[0][0]*B[0][1]+A[0][1]*B[1][1]+A[0][2]*B[2][1], A[0][0]*B[0][2]+A[0][1]*B[1][2]+A[0][2]*B[2][2]),
        (A[1][0]*B[0][0]+A[1][1]*B[1][0]+A[1][2]*B[2][0], A[1][0]*B[0][1]+A[1][1]*B[1][1]+A[1][2]*B[2][1], A[1][0]*B[0][2]+A[1][1]*B[1][2]+A[1][2]*B[2][2]),
        (A[2][0]*B[0][0]+A[2][1]*B[1][0]+A[2][2]*B[2][0], A[2][0]*B[0][1]+A[2][1]*B[1][1]+A[2][2]*B[2][1], A[2][0]*B[0][2]+A[2][1]*B[1][2]+A[2][2]*B[2][2])
    )

def _rot_x(theta):
    c, s = math.cos(theta), math.sin(theta)
    return ((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c))

def _rot_y(theta):
    c, s = math.cos(theta), math.sin(theta)
    return ((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c))

def _rot_z(theta):
    c, s = math.cos(theta), math.sin(theta)
    return ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))

def _axis_rotation(axis_label: str, angle: float):
    if axis_label == 'neg_z': return _rot_z(-angle)
    if axis_label == 'neg_y': return _rot_y(-angle)
    if axis_label == 'neg_x': return _rot_x(-angle)
    if axis_label == 'pos_y': return _rot_y(angle)
    raise ValueError(f"Unknown axis: {axis_label}")

# ============================================================================
# Forward Kinematics
# ============================================================================

def fk(joint_angles) -> Tuple[float, float, float]:
    """Returns (x, y, z) position of the end-effector."""
    if len(joint_angles) != 6:
        raise ValueError(f"Expected 6 joint angles")

    # structural 180° yaw
    R = _rot_z(math.pi)
    pos = _OFFSETS[0]
    
    # Base to first revolute
    R = _mat_mul_mat(R, _axis_rotation(JOINT_AXES[0], joint_angles[0]))

    # Rest of revolutes
    for i in range(1, 6):
        pos = _add(pos, _matmul(R, _OFFSETS[i]))
        R = _mat_mul_mat(R, _axis_rotation(JOINT_AXES[i], joint_angles[i]))

    # End-effector offset
    pos = _add(pos, _matmul(R, _OFFSETS[6]))
    return (float(pos[0]), float(pos[1]), float(pos[2]))


def fk_full(joint_angles):
    """Returns (pos, R) where pos is (x,y,z) and R is 3x3 rotation matrix."""
    if len(joint_angles) != 6:
        raise ValueError(f"Expected 6 joint angles")
    R = _rot_z(math.pi)
    pos = _OFFSETS[0]
    R = _mat_mul_mat(R, _axis_rotation(JOINT_AXES[0], joint_angles[0]))
    for i in range(1, 6):
        pos = _add(pos, _matmul(R, _OFFSETS[i]))
        R = _mat_mul_mat(R, _axis_rotation(JOINT_AXES[i], joint_angles[i]))
    pos = _add(pos, _matmul(R, _OFFSETS[6]))
    return pos, R


if __name__ == '__main__':
    home = [0.0] * 6
    x, y, z = fk(home)
    print(f"FK at home: x={x:.4f}, y={y:.4f}, z={z:.4f}")
