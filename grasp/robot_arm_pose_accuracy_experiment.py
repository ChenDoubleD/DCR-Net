"""Robot-arm pose accuracy experiment script.

The script repeatedly applies random offsets along selected Cartesian/RPY axes,
uses the vision module to estimate the resulting pose, and records the measured
value and error into an Excel workbook.
"""

from __future__ import annotations

import ctypes
import math
import os
import random
import time
from pathlib import Path
from typing import Optional, Sequence, Tuple

# Avoid duplicated OpenMP runtime errors when OpenCV/Ultralytics are loaded.
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import cv2
from openpyxl import Workbook, load_workbook
from models import YOLO

from ik_move_by_offset import ik_move_by_offset_rad_simple
from vision import detect_pose_with_multiframe_fusion, initialize_camera


# -----------------------------------------------------------------------------
# Experiment configuration
# -----------------------------------------------------------------------------
AXES = ["Px", "Py", "Pz", "Rx", "Ry", "Rz"]

EXPERIMENTS_PER_AXIS = 20

# Translation offsets are sampled in meters.
TRANSLATION_RANGE_M = (-0.03, 0.05)  # -30mm~50mm

# Rotation offsets are sampled in radians.
ROTATION_RANGE_RAD = (-10.0 * math.pi / 180.0, 10.0 * math.pi / 180.0)  # -5°~5°

# Baseline values from real measurements.
# Translation axes use millimeters; rotation axes use degrees.
BASELINES = {
    "Px": -1.992,
    "Py": 0.015,
    "Pz": -41.8,
    "Rx": 1.052,
    "Ry": -0.785,
    "Rz": -0.034,
}

EXCEL_PATH = Path("experiment.xlsx")
MODEL_PATH = Path(".")

CAMERA_INDEX = 0


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
INITIAL_JOINTS = (-1.0, 18.25, 67.6, -2.5, 92.0, 270.0)


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


def initialize_arm(initial_joints: Sequence[float] = INITIAL_JOINTS) -> Tuple[int, FloatJoint6]:
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

    return socket_id, joint_target


def move_to_initial_pose(socket_id: int, joint_target: FloatJoint6) -> None:
    """Move the robot arm back to the predefined initial joint pose."""

    ROBOT_DLL.Movej_Cmd(
        socket_id,
        joint_target,
        MOVE_SPEED,
        MOVE_BLEND_RADIUS,
        WAIT_UNTIL_FINISHED,
    )


# -----------------------------------------------------------------------------
# Vision result conversion
# -----------------------------------------------------------------------------
def compute_axis_value(position_m: Sequence[float], rotation_rad: Sequence[float], axis: str) -> float:
    """Convert the raw vision output into the measurement value of one axis.

    Args:
        position_m: Estimated position vector in meters.
        rotation_rad: Estimated rotation vector in radians.
        axis: Target measurement axis, one of Px/Py/Pz/Rx/Ry/Rz.

    Returns:
        Translation value in millimeters or rotation value in degrees.
    """

    if axis == "Px":
        return (-position_m[1] - 0.010) * 1000.0
    if axis == "Py":
        return (-position_m[0] + 0.021) * 1000.0
    if axis == "Pz":
        return (-position_m[2] + 0.193) * 1000.0
    if axis == "Rx":
        return math.degrees(-rotation_rad[0])
    if axis == "Ry":
        return math.degrees(rotation_rad[1])
    if axis == "Rz":
        return math.degrees(-rotation_rad[2])

    raise ValueError(f"Unsupported axis: {axis}")


def generate_random_offset(axis: str) -> Tuple[float, list[float], list[float]]:
    """Generate one random motion command for the selected axis.

    Returns:
        move_display: Move value shown in Excel, in mm or degrees.
        offset_position: Cartesian offset [Px, Py, Pz] in meters.
        offset_rotation: Rotation offset in the IK order [Ry, Rx, Rz], in radians.
    """

    offset_position = [0.0, 0.0, 0.0]
    offset_rotation = [0.0, 0.0, 0.0]

    if axis.startswith("P"):
        move_m = random.uniform(*TRANSLATION_RANGE_M)
        move_display = move_m * 1000.0
        position_index = {"Px": 0, "Py": 1, "Pz": 2}[axis]
        offset_position[position_index] = move_m
        return move_display, offset_position, offset_rotation

    move_rad = random.uniform(*ROTATION_RANGE_RAD)
    move_display = math.degrees(move_rad)

    # The IK helper expects rotation input in the order [Ry, Rx, Rz].
    rotation_index = {"Ry": 0, "Rx": 1, "Rz": 2}[axis]
    offset_rotation[rotation_index] = move_rad

    return move_display, offset_position, offset_rotation


# -----------------------------------------------------------------------------
# Excel utilities
# -----------------------------------------------------------------------------
def load_or_create_workbook(path: Path) -> Workbook:
    """Load an Excel workbook if it exists; otherwise create a new one."""

    if path.exists():
        return load_workbook(path)

    workbook = Workbook()
    default_sheet = workbook.active
    workbook.remove(default_sheet)
    return workbook


def create_sheet_if_missing(workbook: Workbook, sheet_name: str, baseline_value: float):
    """Create an axis sheet and write the header/baseline row if it is missing."""

    if sheet_name in workbook.sheetnames:
        return workbook[sheet_name]

    worksheet = workbook.create_sheet(sheet_name)

    worksheet.cell(row=1, column=1, value="Trial")
    worksheet.cell(row=1, column=2, value="Move")
    worksheet.cell(row=1, column=3, value=sheet_name)
    worksheet.cell(row=1, column=4, value=f"d{sheet_name}")

    worksheet.cell(row=2, column=1, value="Baseline")
    worksheet.cell(row=2, column=3, value=baseline_value)

    worksheet.column_dimensions["A"].width = 12
    worksheet.column_dimensions["B"].width = 12
    worksheet.column_dimensions["C"].width = 14
    worksheet.column_dimensions["D"].width = 14

    print(f"  Created worksheet: {sheet_name}")
    return worksheet


def append_experiment_row(
    sheet_name: str,
    trial_index: int,
    move_value: float,
    measured_value: Optional[float],
    error_value: Optional[float],
) -> None:
    """Append one experiment result row into the target worksheet."""

    workbook = load_or_create_workbook(EXCEL_PATH)
    worksheet = create_sheet_if_missing(workbook, sheet_name, BASELINES.get(sheet_name, 0.0))

    # Data rows start from row 3 because row 1 is the header and row 2 is the baseline.
    row_index = 3
    while worksheet.cell(row=row_index, column=1).value is not None:
        row_index += 1

    worksheet.cell(row=row_index, column=1, value=trial_index)
    worksheet.cell(row=row_index, column=2, value=round(move_value, 4))

    if measured_value is not None:
        worksheet.cell(row=row_index, column=3, value=round(measured_value, 4))
    if error_value is not None:
        worksheet.cell(row=row_index, column=4, value=round(error_value, 4))

    workbook.save(EXCEL_PATH)
    print(f"  [{sheet_name}] Row {row_index} saved.")


# -----------------------------------------------------------------------------
# Single experiment procedure
# -----------------------------------------------------------------------------
def run_single_experiment(
    axis: str,
    trial_index: int,
    socket_id: int,
    joint_target: FloatJoint6,
    pipeline,
    align,
    depth_intrinsics,
    model: YOLO,
) -> bool:
    """Run one movement-detection-recording experiment.

    Returns:
        True if the vision measurement succeeds; otherwise False.
    """

    # 1. Return the arm to the initial pose.
    move_to_initial_pose(socket_id, joint_target)
    time.sleep(1.5)

    # 2. Generate and apply a random motion command.
    move_display, offset_position, offset_rotation = generate_random_offset(axis)
    unit = "mm" if axis.startswith("P") else "deg"
    print(f"  [Axis {axis}] Random move: {move_display:.3f} {unit}")

    ik_move_by_offset_rad_simple(offset_position, offset_rotation)
    time.sleep(0.8)

    # 3. Estimate the target pose using multi-frame visual fusion.
    try:
        position_m, rotation_rad = detect_pose_with_multiframe_fusion(
            pipeline,
            align,
            depth_intrinsics,
            model,
            [0, 0, 0],
        )
    except Exception as exc:
        print(f"  Vision detection error: {exc}")
        position_m, rotation_rad = None, None

    # 4. Convert the vision result and compute the measurement error.
    if position_m is None or rotation_rad is None:
        print("  Vision detection failed. The measured value is left blank.")
        measured_value = None
        error_value = None
    else:
        measured_value = compute_axis_value(position_m, rotation_rad, axis)
        baseline = BASELINES[axis]

        # The sign convention follows the original script: ideal = baseline - move.
        ideal_value = baseline - move_display
        error_value = measured_value - ideal_value

        print(
            f"  Measured: {measured_value:.3f} {unit}, "
            f"Ideal: {ideal_value:.3f}, Error: {error_value:.3f}"
        )

    # 5. Save the experiment result.
    append_experiment_row(axis, trial_index, move_display, measured_value, error_value)
    return measured_value is not None


# -----------------------------------------------------------------------------
# Main entry
# -----------------------------------------------------------------------------
def main() -> None:
    """Run all configured pose accuracy experiments."""

    pipeline = None
    total_success = 0
    total_trials = len(AXES) * EXPERIMENTS_PER_AXIS

    try:
        print("Initializing robot arm...")
        socket_id, joint_target = initialize_arm()

        print("Initializing camera...")
        pipeline, align, depth_intrinsics = initialize_camera(CAMERA_INDEX)

        print("Loading YOLO model...")
        model = YOLO(str(MODEL_PATH))

        for axis in AXES:
            print(f"\n{'=' * 50}\n  Start testing axis: {axis}\n{'=' * 50}")

            for trial_index in range(1, EXPERIMENTS_PER_AXIS + 1):
                print(f"\n--- {axis} trial {trial_index}/{EXPERIMENTS_PER_AXIS} ---")

                try:
                    success = run_single_experiment(
                        axis=axis,
                        trial_index=trial_index,
                        socket_id=socket_id,
                        joint_target=joint_target,
                        pipeline=pipeline,
                        align=align,
                        depth_intrinsics=depth_intrinsics,
                        model=model,
                    )
                    if success:
                        total_success += 1
                except Exception as exc:
                    print(f"  Experiment error: {exc}")
                    append_experiment_row(axis, trial_index, 0.0, None, None)

    finally:
        if pipeline is not None:
            pipeline.stop()
        cv2.destroyAllWindows()
        print(f"\nAll experiments finished. Success: {total_success}/{total_trials}")


if __name__ == "__main__":
    main()
