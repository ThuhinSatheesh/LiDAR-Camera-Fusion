import numpy as np

def convert_coordinate_system(points):
    """Convert right-hand to left-hand coordinate system."""
    points[:, 1] = -points[:, 1]
    points[:, 2] = -points[:, 2]
    return points

def convert_rotation_matrix(rotation_matrix):
    """Adjust rotation matrix for KITTI coordinate system."""
    adjust_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    return np.dot(adjust_matrix, np.dot(rotation_matrix, adjust_matrix.T))

def convert_yaw_angle(angle):
    """Adjust yaw angle for KITTI coordinate system."""
    return -angle