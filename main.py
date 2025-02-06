import os
import glob
import json
import numpy as np
from file_utils import create_output_dirs, save_image, save_point_cloud, save_labels, save_calib_file, generate_calib_data
from coordinate_utils import convert_coordinate_system, convert_rotation_matrix, convert_yaw_angle

#Path to the A2D2 dataset
DATA_PATH = "/media/nvidia/My Passport/camera_lidar_semantic_bboxes_og"

#Output path for the converted KITTI dataset
OUTPUT_PATH = "/media/nvidia/My Passport/camera_lidar_semantic_bboxes_conv"

#TODO combine splitting file as an argument for ComplexYolov4 split format ####################################################

A2D2_TO_KITTI_CLASSES = {
    "Animal": "DontCare",
    "Bicycle": "Cyclist",
    "Bus": "Truck",
    "Car": "Car",
    "CaravanTransporter": "Misc",
    "Cyclist": "Cyclist",
    "EmergencyVehicle": "DontCare",
    "MotorBiker": "Cyclist",
    "Motorcycle": "Cyclist",
    "Pedestrian": "Pedestrian",
    "Trailer": "Truck",
    "Truck": "Truck",
    "UtilityVehicle": "Van",
    "VanSUV": "Car"
}

def process_timestamp_folder(timestamp_folder, calib_json):
    print(f"Processing {timestamp_folder}...")

    if not os.path.isdir(timestamp_folder):
        print(f"Skipping {timestamp_folder}: Not a directory.")
        return

    folder_basename = os.path.basename(timestamp_folder)
    date_str, time_str = folder_basename.split("_")
    timestamp_prefix = date_str + time_str

    camera_dir = os.path.join(timestamp_folder, "camera/cam_front_center")
    label3d_dir = os.path.join(timestamp_folder, "label3D/cam_front_center")
    lidar_dir = os.path.join(timestamp_folder, "lidar/cam_front_center")

    if not (os.path.exists(camera_dir) and os.path.exists(label3d_dir) and os.path.exists(lidar_dir)):
        print(f"Skipping {timestamp_folder}: Missing required directories.")
        return

    image_files = sorted(glob.glob(os.path.join(camera_dir, "*.png")))
    print(f"Found {len(image_files)} images in {camera_dir}.")

    for image_file in image_files:
        base_name = os.path.basename(image_file).replace(".png", "")
        file_id = base_name.split("_")[-1]

        label3d_file = os.path.join(label3d_dir, f"{timestamp_prefix}_label3D_frontcenter_{file_id}.json")
        lidar_file = os.path.join(lidar_dir, f"{timestamp_prefix}_lidar_frontcenter_{file_id}.npz")

        if not os.path.exists(label3d_file) or not os.path.exists(lidar_file):
            continue

        process_sample(file_id, image_file, label3d_file, lidar_file, calib_json)

def process_sample(file_id, image_file, label3d_file, lidar_file, calib_json):
    save_image(file_id, image_file, OUTPUT_PATH)
    lidar_data = np.load(lidar_file)
    save_point_cloud(file_id, lidar_data, OUTPUT_PATH)

    with open(label3d_file, "r") as f:
        label3d_data = json.load(f)

    save_labels(file_id, label3d_data, OUTPUT_PATH, A2D2_TO_KITTI_CLASSES)

    # Generate calibration data
    calib_data = generate_calib_data(calib_json)

    # Save calibration file
    save_calib_file(file_id, OUTPUT_PATH, calib_data)

def main():
    print("Starting A2D2 to KITTI Conversion...")
    create_output_dirs(OUTPUT_PATH)

    # Load calibration JSON once
    with open("/home/nvidia/lidar/a2d2/calib.json", "r") as f:
        calib_json = json.load(f)

    timestamp_folders = sorted(glob.glob(os.path.join(DATA_PATH, "*")))
    for timestamp_folder in timestamp_folders:
        process_timestamp_folder(timestamp_folder, calib_json)

if __name__ == "__main__":
    main()