import os
import shutil
import random

#TODO combine this with convertion script. Add call func()


# Paths
CONVERTED_PATH = "/home/nvidia/lidar/a2d2/conv_kitti"
OUTPUT_PATH = "/home/nvidia/lidar/a2d2/kitti_split"
SPLIT_RATIO = 0.8  # 80% training, 20% testing
VAL_RATIO = 0.1    # 10% of training set for validation

# Create necessary directories
def create_output_dirs(base_path):
    dirs = [
        "training/image_2", "training/velodyne", "training/calib", "training/label_2",
        "testing/image_2", "testing/velodyne", "testing/calib", "testing/label_2",
        "ImageSets"
    ]
    for d in dirs:
        os.makedirs(os.path.join(base_path, d), exist_ok=True)

# Copy files to destination
def copy_files(file_ids, src_path, dest_path):
    for file_id in file_ids:
        # Paths for each subdirectory
        shutil.copy(os.path.join(src_path, "image_2", f"{file_id}.png"), os.path.join(dest_path, "image_2"))
        shutil.copy(os.path.join(src_path, "velodyne", f"{file_id}.bin"), os.path.join(dest_path, "velodyne"))
        shutil.copy(os.path.join(src_path, "calib", f"{file_id}.txt"), os.path.join(dest_path, "calib"))
        shutil.copy(os.path.join(src_path, "label_2", f"{file_id}.txt"), os.path.join(dest_path, "label_2"))

# Main function to split and organize dataset
def organize_dataset():
    # Get all file IDs
    file_ids = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(CONVERTED_PATH, "image_2"))]
    random.shuffle(file_ids)  # Shuffle for randomness

    # Split dataset
    split_idx = int(len(file_ids) * SPLIT_RATIO)
    train_ids = file_ids[:split_idx]
    test_ids = file_ids[split_idx:]

    # Further split training into train and val
    val_split_idx = int(len(train_ids) * VAL_RATIO)
    val_ids = train_ids[:val_split_idx]
    train_ids = train_ids[val_split_idx:]

    # Create output directories
    create_output_dirs(OUTPUT_PATH)

    # Copy files to training, validation, and testing folders
    copy_files(train_ids, CONVERTED_PATH, os.path.join(OUTPUT_PATH, "training"))
    copy_files(val_ids, CONVERTED_PATH, os.path.join(OUTPUT_PATH, "training"))
    copy_files(test_ids, CONVERTED_PATH, os.path.join(OUTPUT_PATH, "testing"))

    # Write ImageSets files
    with open(os.path.join(OUTPUT_PATH, "ImageSets/train.txt"), "w") as f:
        f.writelines([f"{file_id}\n" for file_id in train_ids])
    with open(os.path.join(OUTPUT_PATH, "ImageSets/val.txt"), "w") as f:
        f.writelines([f"{file_id}\n" for file_id in val_ids])
    with open(os.path.join(OUTPUT_PATH, "ImageSets/test.txt"), "w") as f:
        f.writelines([f"{file_id}\n" for file_id in test_ids])

    print("Dataset successfully organized!")

# Run the script
if __name__ == "__main__":
    organize_dataset()