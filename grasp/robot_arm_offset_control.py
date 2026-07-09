"""Manual robot-arm offset control script.

This script initializes the robot arm, moves it to a predefined joint pose,
and then applies a user-defined Cartesian/RPY offset through the inverse
kinematics helper function.
"""

from __future__ import annotations

import ctypes
import os
import time
from typing import Sequence

# Avoid duplicated OpenMP runtime errors when third-party libraries are loaded.
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from ik_move_by_offset import ik_move_by_offset_rad_simple


# -----------------------------------------------------------------------------
# Robot and DLL configuration
# -----------------------------------------------------------------------------
API_VERSION = 65
ROBOT_IP = "192.168.1.100"
ROBOT_PORT = 8080
CONNECT_TIMEOUT_MS = 200

MOVE_SPEED = 20
MOVE_BLEND_RADIUS = 0.0
WAIT_UNTIL_FINISHED = True

CURRENT_DIR = os.path.dirname(__file__)
DLL_PATH = os.path.join(CURRENT_DIR, "RM_Base.dll")
ROBOT_DLL = ctypes.cdll.LoadLibrary(DLL_PATH)

FloatJoint6 = ctypes.c_float * 6

# Initial joint pose. Unit follows the robot SDK convention.
INITIAL_JOINTS = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

# Manual Cartesian offset. Position is in meters; rotation is in radians.
PX_OFFSET = 0.0
PY_OFFSET = 0.0
PZ_OFFSET = 0.0
RX_OFFSET = 0.0
RY_OFFSET = 0.0
RZ_OFFSET = 0.0


# -----------------------------------------------------------------------------
# C structure definitions used by the robot SDK
# -----------------------------------------------------------------------------
class PositionC(ctypes.Structure):
    """Cartesian position structure: x, y, z."""

    _fields_ = [
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("z", ctypes.c_float),
    ]


class QuaternionC(ctypes.Structure):
    """Quaternion orientation structure: w, x, y, z."""

    _fields_ = [
        ("w", ctypes.c_float),
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("z", ctypes.c_float),
    ]


class EulerC(ctypes.Structure):
    """Euler orientation structure: phi, theta, psi."""

    _fields_ = [
        ("phi", ctypes.c_float),
        ("theta", ctypes.c_float),
        ("psi", ctypes.c_float),
    ]


class PoseC(ctypes.Structure):
    """Robot pose structure containing position, quaternion and Euler angles."""

    _fields_ = [
        ("pos", PositionC),
        ("ort", QuaternionC),
        ("eul", EulerC),
    ]


class DeviceMessageC(ctypes.Structure):
    """Device state structure returned by the robot SDK."""

    _fields_ = [
        ("px", ctypes.c_float),
        ("py", ctypes.c_float),
        ("pz", ctypes.c_float),
        ("rx", ctypes.c_float),
        ("ry", ctypes.c_float),
        ("rz", ctypes.c_float),
    ]


# -----------------------------------------------------------------------------
# Robot initialization and motion helpers
# -----------------------------------------------------------------------------
def configure_robot_sdk() -> None:
    """Configure ctypes argument and return types for the robot SDK calls."""

    ROBOT_DLL.Set_Install_Pose.argtypes = (
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_bool,
    )
    ROBOT_DLL.Set_Install_Pose.restype = ctypes.c_int

    ROBOT_DLL.setLwt.argtypes = (ctypes.c_int,)

    ROBOT_DLL.Movej_Cmd.argtypes = (
        ctypes.c_int,
        FloatJoint6,
        ctypes.c_byte,
        ctypes.c_float,
        ctypes.c_bool,
    )
    ROBOT_DLL.Movej_Cmd.restype = ctypes.c_int


def initialize_arm(initial_joints: Sequence[float] = INITIAL_JOINTS) -> int:
    """Initialize the robot API, connect to the arm and move to the start pose."""

    ROBOT_DLL.RM_API_Init(API_VERSION, 0)
    configure_robot_sdk()

    ip_bytes = ROBOT_IP.encode("gbk")
    socket_id = ROBOT_DLL.Arm_Socket_Start(ip_bytes, ROBOT_PORT, CONNECT_TIMEOUT_MS)
    print("Connection status:", socket_id)

    # Set installation pose and DH-related configuration.
    ROBOT_DLL.Set_Install_Pose(socket_id, 0.0, 0.0, 0.0, True)
    ROBOT_DLL.setLwt(0)

    joint_target = FloatJoint6(*initial_joints)
    ROBOT_DLL.Movej_Cmd(
        socket_id,
        joint_target,
        MOVE_SPEED,
        MOVE_BLEND_RADIUS,
        WAIT_UNTIL_FINISHED,
    )
    time.sleep(2.0)

    return socket_id


def apply_manual_offset() -> None:
    """Apply the configured Cartesian and rotation offset to the robot arm."""

    position_offset = [PX_OFFSET, PY_OFFSET, PZ_OFFSET]

    # The IK helper expects rotation input in the order [Ry, Rx, Rz].
    rotation_offset = [RY_OFFSET, RX_OFFSET, RZ_OFFSET]

    print("Position offset [Px, Py, Pz]:", position_offset)
    print("Rotation offset [Ry, Rx, Rz]:", rotation_offset)

    ik_move_by_offset_rad_simple(position_offset, rotation_offset)


def main() -> None:
    """Run the manual offset-control process."""

    initialize_arm()
    time.sleep(2.0)

    try:
        apply_manual_offset()
    finally:
        print("Finished.")


if __name__ == "__main__":
    main()
