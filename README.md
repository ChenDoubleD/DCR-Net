# Grasp-3Signs Visual Pose Detection and Robot Arm Experiment

This repository contains code for training and validating a three-marker visual detection model, together with robot-arm control scripts for pose-offset experiments. The overall workflow is designed for a visual-guided grasping/docking task using three visual signs as pose reference markers.

The codebase includes:

- A customized DCR/DCRM bottleneck module for the detection network.
- YOLO-based training and validation scripts for the `grasp_3signs` dataset.
- A manual robot-arm offset control script.
- A robot-arm pose accuracy experiment script using visual detection and multi-frame pose fusion.

---

## 1. Dataset

The dataset is shared through Baidu Netdisk:

```text
Dataset name: grasp_3signs
Baidu Netdisk link: https://pan.baidu.com/s/1LQzjpaT4gcSDqH3wAqlcbQ
Extraction code: yqbm
```

After downloading and extracting the dataset, place it under the project root or update the dataset path in the YAML file used by the training and validation scripts.

A typical YOLO detection dataset structure is expected:

```text
grasp_3signs/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── Grasp-3signs.yaml
```

The YAML file should define the dataset paths, number of classes, and class names. For this project, the three target signs are expected to correspond to three visual marker classes.

Example YAML format:

```yaml
path: ./grasp_3signs
train: images/train
val: images/val
test: images/test

names:
  0: circle
  1: triangle
  2: square
```

Update the class names if your actual dataset uses different naming.

---

## 2. Project Structure

Recommended file organization:

```text
project_root/
├── README.md
├── train.py
├── val.py
├── dcrm_bottleneck_clean.py
├── robot_arm_offset_control.py
├── robot_arm_pose_accuracy_experiment.py
├── Grasp-3signs.yaml
├── models/
│   └── ...
├── runs/
│   └── ...
├── vision.py
├── ik_move_by_offset.py
├── RM_Base.dll
└── grasp_3signs/
    ├── images/
    └── labels/
```

Main files:

| File                                    | Description                                                                                                          |
| --------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `dcrm_bottleneck_clean.py`              | Cleaned implementation of the DCR/DCRM bottleneck module with English comments and backward-compatible class naming. |
| `train.py`                              | YOLO-based training entry script for the three-sign detection dataset.                                               |
| `val.py`                                | YOLO-based validation/testing script for evaluating trained weights.                                                 |
| `robot_arm_offset_control.py`           | Manual robot-arm initialization and Cartesian/RPY offset control script.                                             |
| `robot_arm_pose_accuracy_experiment.py` | Multi-axis randomized pose accuracy experiment script with visual detection and Excel logging.                       |
| `vision.py`                             | External vision module used for camera initialization and multi-frame pose fusion.                                   |
| `ik_move_by_offset.py`                  | External inverse-kinematics helper used for moving the robot arm by offset.                                          |
| `RM_Base.dll`                           | Robot SDK dynamic library required by the robot control scripts.                                                     |

---

## 3. Environment Requirements

The project is based on Python and PyTorch. A GPU is recommended for model training.

Recommended dependencies:

```bash
pip install torch torchvision torchaudio
pip install ultralytics
pip install opencv-python
pip install numpy
pip install openpyxl
```

Additional requirements:

- A local `models` package compatible with the training and validation scripts.
- Robot SDK file `RM_Base.dll` for hardware control.
- Camera driver and SDK required by `vision.py`.
- CUDA-compatible GPU for training and real-time inference acceleration.

Because the robot and camera scripts depend on external hardware and local SDK files, they should be executed only in the configured experimental environment.

---

## 4. DCR/DCRM Bottleneck Module

The file `dcrm_bottleneck_clean.py` provides a cleaned implementation of the original DCRM bottleneck block. It contains:

- `LayerNormFunction`: custom channel-wise LayerNorm for 4D feature maps.
- `LayerNorm2d`: PyTorch module wrapper for the custom LayerNorm function.
- `SimpleGate`: channel split and element-wise multiplication operation.
- `DepthwiseConv`: depthwise convolution branch with configurable dilation.
- `DCRMBottleneck`: main DCRM/DCR bottleneck implementation.
- `DCRM_bottleneck`: backward-compatible wrapper using the original class name.

The module follows a residual design:

1. Normalize the input feature map.
2. Expand channels using a `1 × 1` convolution.
3. Apply optional local depthwise convolution.
4. Aggregate multi-dilation depthwise branches.
5. Apply SimpleGate-based channel interaction and lightweight channel attention.
6. Project the feature map back to the original channel dimension.
7. Add the residual connection with learnable scaling.

Example usage:

```python
import torch
from dcrm_bottleneck_clean import DCRMBottleneck

x = torch.randn(1, 64, 80, 80)
block = DCRMBottleneck(channels=64, dw_expand=2, dilations=(1, 4, 9))
y = block(x)
print(y.shape)  # torch.Size([1, 64, 80, 80])
```

For compatibility with older code:

```python
from dcrm_bottleneck_clean import DCRM_bottleneck

block = DCRM_bottleneck(c1=64, DW_Expand=2)
```

---

## 5. Training

The training script is `train.py`.

Default training configuration:

```python
model = YOLO(model=r".\models\cfg\models\11\yolo11.yaml", task="detect")
model.train(
    data=r".\Grasp-3signs.yaml",
    imgsz=640,
    epochs=300,
    batch=8,
    workers=0,
    device=0,
    optimizer="SGD",
    close_mosaic=10,
    resume=False,
    single_cls=False,
    cache=False,
    amp=False,
)
```

Run training:

```bash
python train.py
```

Before training, check the following paths:

```python
model = YOLO(model=r".\models\cfg\models\11\yolo11.yaml", task="detect")
data = r".\Grasp-3signs.yaml"
```

Modify them according to your local project structure.

Important parameters:

| Parameter      | Default value | Description                                            |
| -------------- | ------------- | ------------------------------------------------------ |
| `imgsz`        | `640`         | Input image size.                                      |
| `epochs`       | `300`         | Number of training epochs.                             |
| `batch`        | `8`           | Training batch size.                                   |
| `device`       | `0`           | GPU device ID. Use `cpu` if no GPU is available.       |
| `optimizer`    | `SGD`         | Optimizer used for training.                           |
| `close_mosaic` | `10`          | Disable mosaic augmentation during the last 10 epochs. |
| `amp`          | `False`       | Automatic mixed precision is disabled by default.      |

---

## 6. Validation and Testing

The validation script is `val.py`.

Default validation configuration:

```python
model = YOLO(r".\runs\detect/runs/train\Docking20251222-3signs\rtdetrl-tietu+guang\weights\last.pt")
model.val(
    data=r"./data_process/Docking20251222-3signs.yaml",
    imgsz=640,
    batch=16,
    split="test",
    project="runs/val/Docking20251222-3signs",
    name="rtdetrl-tietu+guang-last",
    workers=0,
    device=0,
)
```

Run validation:

```bash
python val.py
```

Before validation, update the following paths:

```python
weights_path = r".\runs\detect/runs/train\Docking20251222-3signs\rtdetrl-tietu+guang\weights\last.pt"
data_yaml = r"./data_process/Docking20251222-3signs.yaml"
```

The default script evaluates the `test` split. Change `split='val'` if you want to evaluate the validation set instead.

---

## 7. Manual Robot Arm Offset Control

The file `robot_arm_offset_control.py` is used for manual robot-arm movement testing.

Main functions:

- Initialize the robot API.
- Connect to the robot arm through IP and port.
- Move the arm to a predefined initial joint pose.
- Apply a manually configured Cartesian and rotation offset.

Key configuration items:

```python
ROBOT_IP = "192.168.1.100"
ROBOT_PORT = 8080
INITIAL_JOINTS = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

PX_OFFSET = 0.0
PY_OFFSET = 0.0
PZ_OFFSET = 0.0
RX_OFFSET = 0.0
RY_OFFSET = 0.0
RZ_OFFSET = 0.0
```

Run the script:

```bash
python robot_arm_offset_control.py
```

Offset convention:

- Position offset: `[Px, Py, Pz]`, in meters.
- Rotation offset: `[Ry, Rx, Rz]`, in radians, following the input order required by `ik_move_by_offset_rad_simple`.

Before running, make sure that:

1. The robot arm is powered on.
2. The robot IP address is correct.
3. `RM_Base.dll` is located in the same directory as the script.
4. `ik_move_by_offset.py` is available.
5. The movement range is safe for the physical setup.

---

## 8. Robot Arm Pose Accuracy Experiment

The file `robot_arm_pose_accuracy_experiment.py` performs randomized pose accuracy experiments. It moves the robot arm along selected axes, estimates the pose using visual detection, compares the result with the ideal value, and saves the error data to Excel.

Main workflow:

1. Initialize the robot arm.
2. Initialize the camera.
3. Load the trained YOLO model.
4. Move the robot to the initial pose.
5. Generate a random translation or rotation offset.
6. Move the robot arm using inverse kinematics.
7. Estimate the target pose using multi-frame visual fusion.
8. Convert the pose result into the selected axis value.
9. Compute the error between the measured value and the ideal value.
10. Save the result to an Excel file.

Key configuration items:

```python
AXES = ["Py", "Rx", "Ry"]
EXPERIMENTS_PER_AXIS = 20
TRANSLATION_RANGE_M = (-0.03, 0.05)
ROTATION_RANGE_RAD = (-10.0 * math.pi / 180.0, 10.0 * math.pi / 180.0)
EXCEL_PATH = Path("DispAM_10.xlsx")
MODEL_PATH = Path(".../weights/best.pt")
CAMERA_INDEX = 0
```

Baseline values:

```python
BASELINES = {
    "Px": -1.992,
    "Py": 0.015,
    "Pz": -41.8,
    "Rx": 1.052,
    "Ry": -0.785,
    "Rz": -0.034,
}
```

Units:

- Translation axes `Px`, `Py`, `Pz`: millimeters.
- Rotation axes `Rx`, `Ry`, `Rz`: degrees.
- Random translation commands are generated in meters and converted to millimeters for Excel output.
- Random rotation commands are generated in radians and converted to degrees for Excel output.

Run the experiment:

```bash
python robot_arm_pose_accuracy_experiment.py
```

Output:

```text
DispAM_10.xlsx
```

The Excel workbook contains one sheet for each tested axis. Each sheet records:

| Column    | Meaning                                       |
| --------- | --------------------------------------------- |
| `Trial`   | Experiment index.                             |
| `Move`    | Commanded movement value.                     |
| Axis name | Measured visual pose value.                   |
| `dAxis`   | Error between measured value and ideal value. |

---

## 9. Notes on Coordinate and Rotation Conventions

The robot experiment code uses specific coordinate and sign conventions inherited from the original experimental setup. In particular:

- The visual measurement is converted to axis-specific values using predefined formulas.
- The ideal value is computed as:

```python
ideal_value = baseline - move_display
```

- The inverse-kinematics helper receives rotation input in the order:

```text
[Ry, Rx, Rz]
```

This order is different from the common `[Rx, Ry, Rz]` representation. Keep this convention unchanged unless the inverse-kinematics function is also modified.

---

## 10. Recommended Usage Order

For a complete experiment, use the following order:

1. Download and extract the dataset.
2. Update `Grasp-3signs.yaml` according to the dataset location.
3. Train the detection model:

```bash
python train.py
```

4. Validate the trained model:

```bash
python val.py
```

5. Update `MODEL_PATH` in `robot_arm_pose_accuracy_experiment.py` to the trained weight file, such as:

```text
runs/train/your_experiment/weights/best.pt
```

6. Test robot-arm offset control in a safe workspace:

```bash
python robot_arm_offset_control.py
```

7. Run pose accuracy experiments:

```bash
python robot_arm_pose_accuracy_experiment.py
```

---

## 11. Common Issues

### 1. `RM_Base.dll` cannot be found

Make sure `RM_Base.dll` is placed in the same directory as the robot control scripts.

### 2. Dataset YAML path is incorrect

Check the `data` argument in `train.py` and `val.py`. The path should point to the correct dataset YAML file.

### 3. Weight file path is incorrect

Update the YOLO weight path in `val.py` or `robot_arm_pose_accuracy_experiment.py` before running validation or robot experiments.

### 4. CUDA device error

If no GPU is available, change:

```python
device = 0
```

to:

```python
device = "cpu"
```

### 5. Camera initialization fails

Check the camera index and verify that the camera SDK used by `vision.py` is correctly installed.

### 6. Robot movement direction is unexpected

Check the coordinate convention, sign definition, and the rotation input order `[Ry, Rx, Rz]` used by the inverse-kinematics helper.

---

## 12. Safety Notice

The robot-control scripts directly command physical hardware. Before execution:

- Verify all offsets and joint poses.
- Keep the robot workspace clear.
- Use low speed during debugging.
- Keep the emergency stop accessible.
- Run manual offset tests before automated randomized experiments.

---

## 13. License

No license is specified. Please add a license file if this project will be shared publicly.
