import os
import shutil
import cv2
import numpy as np
from kitti_writer import write_label_file, write_calib_file
from coordinate_utils import convert_coordinate_system, convert_rotation_matrix, convert_yaw_angle

def create_output_dirs(output_path):
    dirs = ["image_2", "velodyne", "calib", "label_2"]
    for d in dirs:
        os.makedirs(os.path.join(output_path, d), exist_ok=True)

def save_image(file_id, image_file, output_path):
    image = cv2.imread(image_file)
    output_file = os.path.join(output_path, "image_2", f"{file_id}.png")
    cv2.imwrite(output_file, image)

def save_point_cloud(file_id, lidar_data, output_path):
    """
    Save LiDAR point cloud data in KITTI format (.bin).

    Args:
        file_id (str): Unique identifier for the file.
        lidar_data (npz): LiDAR data loaded from .npz file.
        output_path (str): Directory to save the .bin file.
    """
    # Extract points (assume 'points' key exists in the .npz file)
    points = lidar_data["points"]  # Shape: (N, 3) or (N, 4)

    # Check if intensity is present; if not, add default intensity (e.g., all zeros)
    if points.shape[1] == 3:
        intensity = np.zeros((points.shape[0], 1), dtype=np.float32)
        points = np.hstack((points, intensity))  # Shape: (N, 4)

    # Convert coordinate system to KITTI format (left-handed)
    points = convert_coordinate_system(points)

    # Save as .bin file
    output_file = os.path.join(output_path, "velodyne", f"{file_id}.bin")
    points.astype(np.float32).tofile(output_file)


def save_labels(file_id, label3d_data, output_path, class_mapping):
    labels = []
    for box_id, box in label3d_data.items():
        class_name = class_mapping.get(box["class"], None)
        if class_name is None:
            print(f"Warning: Class '{box['class']}' not found in mapping. Defaulting to 'DontCare'.")
            class_name = "DontCare"

        truncation = box.get("truncation", 0.0)
        occlusion = box.get("occlusion", 0.0)
        alpha = box.get("alpha", 0.0)
        bbox_2d = box["2d_bbox"]
        dimensions = box["size"]
        center = box["center"]
        rotation_y = box["rot_angle"]

        # Convert rotation angle to KITTI coordinate system
        rotation_y_kitti = convert_yaw_angle(rotation_y)

        label = {
            "class": class_name,
            "truncation": truncation,
            "occlusion": occlusion,
            "alpha": alpha,
            "2d_bbox": bbox_2d,
            "size": dimensions,
            "center": center,
            "rot_angle": rotation_y_kitti
        }
        labels.append(label)

    write_label_file(file_id, labels, output_path)

def save_calib_file(file_id, output_path, calib_data):
    write_calib_file(file_id, calib_data, output_path)

def generate_intrinsic_matrix(image_width, image_height, hfov_deg, vfov_deg):
    """
    Calculate the intrinsic camera matrix K based on image size and field of view.
    """
    hfov_rad = np.deg2rad(hfov_deg)
    vfov_rad = np.deg2rad(vfov_deg)

    fx = (image_width / 2) / np.tan(hfov_rad / 2)
    fy = (image_height / 2) / np.tan(vfov_rad / 2)
    cx = image_width / 2
    cy = image_height / 2

    K = np.array([
        [fx,  0, cx],
        [0, fy, cy],
        [0,  0,  1]
    ])

    return K

def generate_calib_data(calib_json):
    """
    Generate calibration data for KITTI format using A2D2 calibration JSON.

    Args:
        calib_json (dict): A2D2 calibration JSON data.

    Returns:
        calib_data (dict): Dictionary containing P0-P3, R0_rect, Tr_velo_to_cam.
    """
    cameras = calib_json["cameras"]
    lidars = calib_json["lidars"]

    # Extract intrinsic parameters for the front-center camera (P2)
    fc_cam = cameras["front_center"]
    fc_intrinsic = np.array(fc_cam["CamMatrix"])

    # Extract intrinsic parameters for surround cameras (P0, P1, P3)
    surround_cam_intrinsics = [
        np.array(cameras["front_left"]["CamMatrix"]),
        np.array(cameras["front_right"]["CamMatrix"]),
        np.array(cameras["rear_center"]["CamMatrix"]),
    ]

    # Compute projection matrices
    P2 = np.hstack((fc_intrinsic, np.zeros((3, 1))))
    P0 = np.hstack((surround_cam_intrinsics[0], np.zeros((3, 1))))
    P1 = np.hstack((surround_cam_intrinsics[1], np.zeros((3, 1))))
    P3 = np.hstack((surround_cam_intrinsics[2], np.zeros((3, 1))))

    # Rectification matrix
    R0_rect = np.eye(3)

    # LiDAR-to-camera transformation for front-center LiDAR
    fc_lidar = lidars["front_center"]
    lidar_origin = np.array(fc_lidar["view"]["origin"])
    lidar_x_axis = np.array(fc_lidar["view"]["x-axis"])
    lidar_y_axis = np.array(fc_lidar["view"]["y-axis"])
    lidar_z_axis = np.cross(lidar_x_axis, lidar_y_axis)  # Compute z-axis
    rotation_matrix = np.vstack([lidar_x_axis, lidar_y_axis, lidar_z_axis]).T
    translation_vector = lidar_origin.reshape(3, 1)

    # Assemble Tr_velo_to_cam matrix
    Tr_velo_to_cam = np.hstack((rotation_matrix, translation_vector))

    calib_data = {
        "P0": P0.flatten(),
        "P1": P1.flatten(),
        "P2": P2.flatten(),
        "P3": P3.flatten(),
        "R0_rect": R0_rect.flatten(),
        "Tr_velo_to_cam": Tr_velo_to_cam.flatten(),
    }

    return calib_data