import cv2
import pyrealsense2 as rs
import numpy as np
from models import YOLO
import time
import math


def initialize_camera(camera_id=0):
    """初始化单个相机"""
    pipeline = rs.pipeline()
    config = rs.config()

    # 启用设备（如果指定了相机ID）
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) > camera_id:
        config.enable_device(devices[camera_id].get_info(rs.camera_info.serial_number))
        print(f"相机 {camera_id} 序列号: {devices[camera_id].get_info(rs.camera_info.serial_number)}")

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)

    # 启动流
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    # 创建对齐对象（对齐深度到颜色）
    align_to = rs.stream.color
    align = rs.align(align_to)

    # 获取相机内参
    depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
    depth_intrinsics = depth_profile.get_intrinsics()

    return pipeline, align, depth_intrinsics


def reset_camera_pipeline(pipeline, align, camera_id=0):
    """重置相机管道"""
    try:
        pipeline.stop()
        time.sleep(0.5)  # 短暂等待
    except:
        pass

    # 重新初始化
    return initialize_camera(camera_id)


def detect_pose_with_multiframe_fusion(pipeline, align, depth_intrinsics, model, ee_rpy_exp_0, num_frames=10,
                                       visualization_window="Pose Detection"):
    """
    使用多帧融合检测位姿

    参数:
        pipeline: RealSense管道
        align: 对齐对象
        depth_intrinsics: 深度相机内参
        model: YOLO模型
        num_frames: 融合帧数
        visualization_window: 可视化窗口名称

    返回:
        pos_ET0: L-sign的世界坐标 [x, y, z]
        rpy_ET0: 欧拉角 [roll, pitch, yaw]，其中roll=pitch=0，yaw为计算出的角度
    """

    # 颜色映射用于不同类别
    COLORS = {
        "sign1": (0, 255, 0),  # 绿色
        "sign2": (0, 255, 255),  # 黄色
        "sign3": (255, 255, 0),  # 青色
    }

    def pixel_to_world(x, y, depth_frame):
        """Convert pixel coordinates to world coordinates"""
        depth = depth_frame.get_distance(x, y)
        return rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x, y], depth)

    def calculate_X_rotation_from_world(p1_world, p2_world):
        """
        Calculate X-axis rotation angle from world coordinates
        """
        dy = p2_world[1] - p1_world[1]
        dz = p2_world[2] - p1_world[2]

        # Calculate angle (radians)
        angle_rad = math.atan(dz/abs(dy))

        # Convert to degrees
        angle_deg = math.degrees(angle_rad)

        return angle_rad, angle_deg

    def calculate_Y_rotation_from_world(p1_world, p2_world):
        """
        Calculate Y-axis rotation angle from world coordinates
        """
        dx = p2_world[0] - p1_world[0]
        dz = p2_world[2] - p1_world[2]

        # Calculate angle (radians)
        angle_rad = math.atan(dz/abs(dx))

        # Convert to degrees
        angle_deg = math.degrees(angle_rad)

        return angle_rad, angle_deg

    def calculate_Z_rotation_from_world(p1_world, p2_world):
        """
        Calculate Z-axis rotation angle from world coordinates
        """
        dx = p2_world[0] - p1_world[0]
        dy = p2_world[1] - p1_world[1]

        # Calculate angle (radians)
        angle_rad = math.atan(dy/abs(dx))

        # Convert to degrees
        angle_deg = math.degrees(angle_rad)

        return angle_rad, angle_deg

    def mad_filter(data, threshold=3.0):
        """基于中位数绝对偏差(MAD)的异常点过滤"""
        if len(data) < 3:
            return data  # 数据太少，无法有效过滤

        # 计算中位数
        median = np.median(data, axis=0)

        # 计算绝对偏差
        abs_dev = np.abs(data - median)

        # 计算中位数绝对偏差(MAD)
        mad = np.median(abs_dev, axis=0)

        # 避免除以零
        mad[mad == 0] = 1e-6

        # 计算标准化偏差
        z_scores = 0.6745 * abs_dev / mad  # 0.6745是标准正态分布的分位数

        # 检测异常点
        inliers = []
        for i in range(len(data)):
            if np.all(z_scores[i] < threshold):
                inliers.append(data[i])

        return np.array(inliers)

    def find_central_group(detections_by_class, image_center, max_group_distance=150):
        """
        找到靠近画面中心的一组检测点

        参数:
            detections_by_class: 按类别分组的检测结果
            image_center: 画面中心坐标 (cx, cy)
            max_group_distance: 同一组点之间的最大像素距离阈值

        返回:
            包含sign1, sign2, sign3的字典，表示靠近中心的那一组
        """
        sign1_list = detections_by_class.get("sign1", [])
        sign2_list = detections_by_class.get("sign2", [])
        sign3_list = detections_by_class.get("sign3", [])

        # 如果没有足够的点，返回None
        if len(sign1_list) == 0 or len(sign3_list) == 0:
            return None

        # 计算每个点到画面中心的距离
        def distance_to_center(point):
            px, py, _ = point
            return math.sqrt((px - image_center[0]) ** 2 + (py - image_center[1]) ** 2)

        # 找到离中心最近的sign1和sign3
        closest_sign1 = min(sign1_list, key=distance_to_center)
        closest_sign3 = min(sign3_list, key=distance_to_center)

        # 如果sign2存在，找到离中心最近的sign2
        closest_sign2 = None
        if len(sign2_list) > 0:
            closest_sign2 = min(sign2_list, key=distance_to_center)

        # 检查sign1和sign3之间的距离是否合理（属于同一组）
        distance_sign1_sign3 = math.sqrt(
            (closest_sign1[0] - closest_sign3[0]) ** 2 +
            (closest_sign1[1] - closest_sign3[1]) ** 2
        )

        if distance_sign1_sign3 > max_group_distance:
            # 可能不属于同一组，尝试寻找更匹配的组合
            # 寻找距离sign1最近的sign3
            closest_sign3_to_sign1 = min(sign3_list,
                                         key=lambda p: math.sqrt(
                                             (p[0] - closest_sign1[0]) ** 2 + (p[1] - closest_sign1[1]) ** 2))

            # 寻找距离sign3最近的sign1
            closest_sign1_to_sign3 = min(sign1_list,
                                         key=lambda p: math.sqrt(
                                             (p[0] - closest_sign3[0]) ** 2 + (p[1] - closest_sign3[1]) ** 2))

            # 选择更靠近中心的那一组
            distance1 = distance_to_center(closest_sign1)
            distance2 = distance_to_center(closest_sign1_to_sign3)

            if distance1 < distance2:
                # 以sign1为中心
                selected_sign1 = closest_sign1
                selected_sign3 = closest_sign3_to_sign1
            else:
                # 以sign3为中心
                selected_sign1 = closest_sign1_to_sign3
                selected_sign3 = closest_sign3

            # 如果有sign2，选择距离中心最近的sign2
            selected_sign2 = closest_sign2
        else:
            # 最近的sign1和sign3属于同一组
            selected_sign1 = closest_sign1
            selected_sign3 = closest_sign3
            selected_sign2 = closest_sign2

        return {
            "sign1": selected_sign1,
            "sign2": selected_sign2,
            "sign3": selected_sign3
        }

    def process_multiframe_detection(pipeline, align, model, num_frames):
        """多帧融合检测"""
        valid_keypoints = []
        all_frames_data = []

        for frame_idx in range(num_frames):
            # 等待帧
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                time.sleep(0.1)
                continue

            # 转换为numpy数组
            color_image = np.asanyarray(color_frame.get_data())

            image_center = (color_image.shape[1] // 2, color_image.shape[0] // 2)

            # YOLO目标检测
            results = model(color_image, conf=0.5, iou=0.5, verbose=False, device=0)

            # 解析检测结果 - 收集所有检测点
            detections_by_class = {"sign1": [], "sign2": [], "sign3": []}

            for result in results:
                for box in result.boxes:
                    # Get bounding box coordinates and confidence
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf.item()
                    cls = int(box.cls.item())
                    class_name = result.names[cls]

                    # Calculate center point
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    # 针对画面中有多组标志点的情况-添加到对应类别的列表
                    if class_name in detections_by_class:
                        detections_by_class[class_name].append((cx, cy, conf))

            # 针对画面中有多组标志点的情况-找到靠近画面中心的那一组
            central_group = find_central_group(detections_by_class, image_center)

            if (central_group is None or central_group["sign1"] is None or central_group["sign2"] is None
                    or central_group["sign3"] is None):
                print(f"Frame {frame_idx + 1}: No valid central group found")
                continue

            A_pixel = central_group["sign1"]
            B_pixel = central_group["sign2"]
            C_pixel = central_group["sign3"]

            # 检查是否检测到所有点
            if A_pixel is not None and B_pixel is not None and C_pixel is not None:
                # 转换为世界坐标
                try:
                    A_world = pixel_to_world(A_pixel[0], A_pixel[1], depth_frame)
                    B_world = pixel_to_world(B_pixel[0], B_pixel[1], depth_frame)
                    C_world = pixel_to_world(C_pixel[0], C_pixel[1], depth_frame)

                    # 检查深度有效性
                    if (A_world is not None and B_world is not None and C_world is not None and
                            all(coord != 0 for coord in A_world) and
                            all(coord != 0 for coord in B_world) and
                            all(coord != 0 for coord in C_world)):
                        valid_keypoints.append((A_world, B_world, C_world))
                        all_frames_data.append((color_image, depth_frame, A_pixel, B_pixel, C_pixel))

                except Exception as e:
                    print(f"Frame {frame_idx + 1}: Error in coordinate conversion - {e}")
                    continue

            # 显示当前帧检测状态
            display_img = color_image.copy()
            status_text = f"Frame {frame_idx + 1}/{num_frames} - Points: {len(valid_keypoints)}"
            cv2.putText(display_img, status_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # 可视化当前帧检测结果
            if A_pixel is not None and B_pixel is not None and C_pixel is not None:
                points = [(A_pixel[0], A_pixel[1]), (B_pixel[0], B_pixel[1]), (C_pixel[0], C_pixel[1])]
                colors = [(0, 255, 0), (255, 255, 0), (0, 255, 255)]
                labels = ['A', 'B', 'C']

                for i, (x, y) in enumerate(points):
                    cv2.circle(display_img, (int(x), int(y)), 8, colors[i], -1)
                    cv2.putText(display_img, labels[i], (int(x) + 10, int(y) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # # 绘制连线
                # cv2.line(display_img,
                #          (int(A_pixel[0]), int(A_pixel[1])),
                #          (int(B_pixel[0]), int(B_pixel[1])),
                #          (255, 255, 0), 2)

            cv2.imshow(visualization_window, display_img)
            cv2.waitKey(1)
            time.sleep(0.1)  # 短暂延迟以确保可视化

        return valid_keypoints, all_frames_data

    def filter_and_calculate_median(keypoints_list):
        """过滤异常关键点并计算中值"""
        if not keypoints_list:
            return None, None, None

        # 分别提取A、B、C点的坐标
        A_points = np.array([kp[0] for kp in keypoints_list])
        B_points = np.array([kp[1] for kp in keypoints_list])
        C_points = np.array([kp[2] for kp in keypoints_list])

        # 过滤异常点
        filtered_A = mad_filter(A_points)
        filtered_B = mad_filter(B_points)
        filtered_C = mad_filter(C_points)

        # 确保过滤后仍有足够的数据点
        if len(filtered_A) < 1 or len(filtered_B) < 1 or len(filtered_C) < 1:
            print(f"Warning: Not enough points after filtering")
            # 使用所有点计算中值
            if len(A_points) > 0:
                A_median = np.median(A_points, axis=0)
            else:
                return None, None, None

            if len(B_points) > 0:
                B_median = np.median(B_points, axis=0)
            else:
                return None, None, None

            if len(C_points) > 0:
                C_median = np.median(C_points, axis=0)
            else:
                return None, None, None

        else:
            # 计算过滤后的中值
            A_median = np.median(filtered_A, axis=0)
            B_median = np.median(filtered_B, axis=0)
            C_median = np.median(filtered_C, axis=0)

        return A_median, B_median, C_median

    # 主处理逻辑
    print(f"Starting multi-frame detection with {num_frames} frames...")

    # 多帧融合检测
    valid_keypoints, all_frames_data = process_multiframe_detection(pipeline, align, model, num_frames)

    if not valid_keypoints:
        print("No valid keypoints detected in any frame.")
        return None, None

    print(f"Successfully processed {len(valid_keypoints)} valid frames out of {num_frames}")

    # 过滤关键点并计算中值
    A_median, B_median, C_median = filter_and_calculate_median(valid_keypoints)

    if A_median is None or B_median is None or C_median is None:
        print("Failed to calculate median keypoints.")
        return None, None

    print(f"Filtered keypoints - A: {A_median}, B: {B_median}, C: {C_median}")

    # 计算角度（基于世界坐标/像素坐标）
    try:
        world_angle_rad_X, world_angle_de_X = calculate_Y_rotation_from_world(B_median, C_median)
        world_angle_rad_Y, world_angle_de_Y = calculate_X_rotation_from_world(A_median, B_median)
        world_angle_rad_Z, world_angle_de_Z = calculate_Z_rotation_from_world(B_median, C_median)

        # print(f"X angle: {world_angle_de_X:.2f} deg ({world_angle_rad_X:.3f} rad)")
        # print(f"Y angle: {world_angle_de_Y:.2f} deg ({world_angle_rad_Y:.3f} rad)")
        # print(f"Z angle: {world_angle_de_Z:.2f} deg ({world_angle_rad_Z:.3f} rad)")

    except Exception as e:
        print(f"Error calculating angle from world coordinates: {e}")
        return None, None

    # 最终可视化
    if all_frames_data:
        last_frame_data = all_frames_data[-1]
        color_image, _, A_pixel, B_pixel, C_pixel = last_frame_data

        display_img = color_image.copy()

        # 绘制关键点
        points = [(A_pixel[0], A_pixel[1]), (B_pixel[0], B_pixel[1]), (C_pixel[0], C_pixel[1])]
        colors = [(0, 255, 0), (255, 255, 0), (0, 255, 255)]  # A, B, C点的颜色
        labels = ['A', 'B', 'C']

        for i, (x, y) in enumerate(points):
            cv2.circle(display_img, (int(x), int(y)), 8, colors[i], -1)
            cv2.putText(display_img, labels[i], (int(x) + 10, int(y) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # # 绘制连线
        # cv2.line(display_img,
        #          (int(A_pixel[0]), int(A_pixel[1])),
        #          (int(B_pixel[0]), int(B_pixel[1])),
        #          (255, 255, 0), 3)

        # 显示角度信息
        y_offset = 30
        cv2.putText(display_img, f"Position: [{B_median[0]:.3f}, {B_median[1]:.3f}, {B_median[2]:.3f}]",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.putText(display_img, f"Yaw: {world_angle_de_Z:.2f} deg ({world_angle_rad_Z:.3f} rad)",
                    (10, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(display_img, f"pitch: {world_angle_de_Y:.2f} deg ({world_angle_rad_Y:.3f} rad)",
                    (10, y_offset + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(display_img, f"roll: {world_angle_de_X:.2f} deg ({world_angle_rad_X:.3f} rad)",
                    (10, y_offset + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(display_img, f"Frames used: {len(valid_keypoints)}/{num_frames}",
                    (10, y_offset + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display_img, "Press any key to continue...",
                    (10, y_offset + 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow(visualization_window, display_img)
        # cv2.waitKey(0)

    # 返回位姿信息
    pos_ET0 = B_median.tolist()
    rpy_ET0 = [world_angle_rad_X, world_angle_rad_Y, world_angle_rad_Z]

    return pos_ET0, rpy_ET0


# 使用示例
if __name__ == "__main__":
    # 初始化RealSense管道
    pipeline = rs.pipeline()
    config = rs.config()
    camera_id = 0

    # 启用设备（如果指定了相机ID）
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) > camera_id:
        config.enable_device(devices[camera_id].get_info(rs.camera_info.serial_number))
        print(f"相机 {camera_id} 序列号: {devices[camera_id].get_info(rs.camera_info.serial_number)}")

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)

    # 启动流
    profile = pipeline.start(config)

    # 创建对齐对象（对齐深度到颜色）
    align_to = rs.stream.color
    align = rs.align(align_to)

    # 获取相机内参
    depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
    depth_intrinsics = depth_profile.get_intrinsics()

    # 加载YOLOv8模型
    model = YOLO(r'E:\code\Docking-raw\runs\train\Docking20251222-3signs-tietu+guang\yolov11n-DispAMs\yolov11n-DispAM\weights\best.pt')

    try:
        # 第一次检测
        print("=== First Detection ===")
        pos1, rpy1 = detect_pose_with_multiframe_fusion(
            pipeline, align, depth_intrinsics, model, [0, 0, 0],
            num_frames=10, visualization_window="Detection 1"
        )

        if pos1 is not None and rpy1 is not None:
            print(f"Position: {pos1}")
            print(f"RPY: {rpy1}")

        # 第二次检测（新的可视化窗口）
        print("\n=== Second Detection ===")
        pos2, rpy2 = detect_pose_with_multiframe_fusion(
            pipeline, align, depth_intrinsics, model, [0, 0, 0],
            num_frames=10, visualization_window="Detection 2"
        )

        if pos2 is not None and rpy2 is not None:
            print(f"Position: {pos2}")
            print(f"RPY: {rpy2}")

    finally:
        # 清理资源
        pipeline.stop()
        cv2.destroyAllWindows()