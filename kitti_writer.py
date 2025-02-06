import os
import numpy as np

def write_label_file(file_id, labels, output_path):
    """Write labels to a KITTI format label file."""
    label_file_path = os.path.join(output_path, "label_2", f"{file_id}.txt")
    with open(label_file_path, "w") as f:
        for label in labels:
            class_name = label["class"]
            truncation = label.get("truncation", 0.0)
            occlusion = label.get("occlusion", 0)
            alpha = label.get("alpha", 0.0)
            bbox = label["2d_bbox"]
            dimensions = label["size"]
            center = label["center"]
            rotation_y = label["rot_angle"]

            line = f"{class_name} {truncation} {occlusion} {alpha} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]} {dimensions[2]} {dimensions[1]} {dimensions[0]} {center[0]} {center[1]} {center[2]} {rotation_y}\n"
            f.write(line)

def write_calib_file(file_id, calib_data, output_path):
    """Write calibration data to a KITTI format calibration file."""
    calib_file_path = os.path.join(output_path, "calib", f"{file_id}.txt")
    with open(calib_file_path, "w") as f:
        # Projection matrices P0 to P3
        for i, P_key in enumerate(['P0', 'P1', 'P2', 'P3']):
            P = calib_data[P_key]
            P_line = ' '.join([f"{num:.6e}" for num in P])
            f.write(f"{P_key}: {P_line}\n")

        # Rectification matrix R0_rect
        R0_rect = calib_data['R0_rect']
        R0_rect_line = ' '.join([f"{num:.6e}" for num in R0_rect])
        f.write(f"R0_rect: {R0_rect_line}\n")

        # Transformation matrix Tr_velo_to_cam
        Tr_velo_to_cam = calib_data['Tr_velo_to_cam']
        Tr_line = ' '.join([f"{num:.6e}" for num in Tr_velo_to_cam])
        f.write(f"Tr_velo_to_cam: {Tr_line}\n")