""" Module with MOTS2020 data reader helper functions """
import cv2
import numpy as np

# from PIL import Image
from scipy.spatial.transform import Rotation as R

from mots_tracker import utils


def id2imgpath(seq_idx, img_idx, input_dir):
    """creates path from image id
    Args:
        seq_idx (str): sequence id
        img_idx (str): image id
        input_dir (Path): path to the folder with the images
    Returns:
        str: path to the image
    """
    img_name = img_idx + ".jpg"
    img_path = input_dir / seq_idx / "img1" / img_name
    return str(img_path)


def id2depthpath(seq_idx, depth_idx, input_dir, depth_folder):
    """creates path from depth image id
    Args:
        seq_idx (str): sequence id
        img_idx (str): segmentation image id
        input_dir (Path): path to the folder with the segmentation images
    Returns:
        str: path to the depth image
    """
    depth_name = depth_idx + ".npy"
    depth_path = input_dir / seq_idx / depth_folder / depth_name
    return str(depth_path)


def path2id(path):
    """extracts image name from absolute path
    Args:
        path (Path): absolute path to a file
    Returns:
        str: name of the file
    """
    str_path = str(path).split("/")[-1]
    return str_path.split(".")[0]


def read_file_names(path):
    """read all file names in a given path
    Args:
        path (Path): absolute path to a directory
    Returns:
        list(str): file names
    """
    return [str(file_name) for file_name in path.glob("**/*")]


def load_motsynth_depth_image(img_path, shape=None, max_depth=50):
    """Load depth image from .png file
    Args:
        img_path (str): path to the image
    Returns:
        ndarray: depth map
    """
    n = 1.04187
    f = 800
    abs_min = 1008334389
    abs_max = 1067424357
    depth = cv2.imread(img_path)[:, :, 0]
    depth = np.uint32(depth)
    depth = depth / 255
    depth = (depth * (abs_max - abs_min)) + abs_min
    depth = depth.astype("uint32")
    depth.dtype = "float32"
    y = (-(n * f) / (n - f)) / (depth - (n / (n - f)))
    y = y.reshape((1080, 1920))
    # y[y > max_depth] = 0
    return y


def read_kitti_bb_file(path):
    """Reads bounding box KITTI file
    Args:
        path (str): path to the ground truth file
    Returns:
        bb_data (ndarray): array in [frame_id, ped_id, x1, y1, x2, y2, obj_type] format
    """
    bb_file = open(path, "r")
    bb_lines = bb_file.readlines()
    bb_data = []
    obj_types = []
    for i, line in enumerate(bb_lines):
        parts = line.split(" ")
        obj_type = parts[2]
        if obj_type not in ["Car", "Pedestrian"]:
            continue
        obj_types.append(obj_type)
        frame_id, ped_id = parts[:2]
        x1, y1, x2, y2 = parts[6:10]
        bb_data.append([frame_id, ped_id, x1, y1, x2, y2])
    bb_data = np.array(bb_data, dtype=np.float64)
    bb_file.close()
    return bb_data, obj_types


def read_kitti_calib(path):
    """Reads KITTI calibration file
    Args:
        path (str): path to the calibration file
    Returns:
        params (list): intrinsics and rectification matrix
        consists of 3x4 intrinsics and 3x3 rectification rotation
        see detailed information here:
        https://www.mrt.kit.edu/z/publ/download/2013/GeigerAl2013IJRR.pdf
    """
    calib_file = open(path, "r")
    calib_lines = [line.strip() for line in calib_file.readlines()]
    params = []
    for line in sorted(calib_lines):
        parts = line.split(" ")
        if parts[0] not in ["P2:", "R_rect"]:
            continue
        params.append(np.array(parts[1:], dtype=np.float64).reshape((3, -1)))
    calib_file.close()
    return params


def read_mot_bb_file(path):
    """Reads bounding box mot file
    Args:
        path (str): path to the ground truth file
    Returns:
        bb_data (ndarray): array in [frame_id, ped_id, x, y, w, h] format
    """
    bb_file = open(path, "r")
    bb_lines = bb_file.readlines()
    bb_data = np.zeros(
        (len(bb_lines), 6), dtype=np.float64
    )  # all coordinates are integers
    for i, line in enumerate(bb_lines):
        frame_id, ped_id, x, y, w, h = line.split(",")[:6]
        bb_data[i, ...] = np.array([frame_id, ped_id, x, y, w, h], dtype=np.float64)
    bb_file.close()
    return bb_data


def read_mot_seg_file(path):
    """Reads segmentation mot file
    Args:
        path (str): path to the ground truth file
    Returns:
        seg_data (ndarray): array in [frame_id, ped_id, h, w] format
        mask_strings (list): list of raw coco masks strings
    """
    seg_file = open(path, "r")
    seg_lines = seg_file.readlines()
    seg_data = np.zeros((len(seg_lines), 4), dtype=np.float64)
    mask_strings = [None] * len(seg_lines)
    for i, line in enumerate(seg_lines):
        split_line = line.split(" ")
        # here, ped_id can be either pedestrian id in case of gt data
        # or a confidence score of a predictor
        if len(split_line) == 10:  # trackrcnn public detections
            frame_id, _, _, _, _, ped_id, _, height, width, mask_string = split_line
        elif len(split_line) == 5:  # detectron2 detections
            frame_id, ped_id, height, width, mask_string = split_line
        else:
            frame_id, ped_id, _, height, width, mask_string = split_line
        seg_data[i, ...] = np.array([frame_id, ped_id, height, width], dtype=np.float64)
        mask_strings[i] = mask_string.strip()
    seg_file.close()
    return seg_data, mask_strings


def read_motsynth_2d_keypoints_file(path: str, keypoints_num=22) -> np.ndarray:
    """Reads motsynth 2D keypoints file
    Args:
        path (str): path to the ground truth file
        keypoints_num (int): number of keypoints per body
    Returns:
        keypoints_2d: integer array of shape Nx3
    """
    ktp_file = open(path, "r")
    kpt_lines = ktp_file.readlines()
    kpt_2d_data = np.zeros(
        (len(kpt_lines), keypoints_num * 3 + 2), dtype=np.int
    )  # in the pixel space, hence integers
    for i, line in enumerate(kpt_lines):
        kpt_2d_data[i, ...] = np.array(line.split(","), dtype=np.int)
    ktp_file.close()
    return kpt_2d_data


def read_motsynth_3d_keypoints_file(path: str, keypoints_num=22) -> np.ndarray:
    """Reads motsynth 3D keypoints file
    Args:
        path (str): path to the ground truth file
        keypoints_num (int): number of keypoints per body
    Returns:
        keypoints_3d: integer array of shape Nx4
    """
    ktp_file = open(path, "r")
    kpt_lines = ktp_file.readlines()
    kpt_3d_data = np.zeros(
        (len(kpt_lines), keypoints_num * 4 + 2), dtype=np.float32
    )  # in the real world space, hence float
    for i, line in enumerate(kpt_lines):
        kpt_3d_data[i, ...] = np.array(line.split(","), dtype=np.float32)
    ktp_file.close()
    return kpt_3d_data


def read_motsynth_egomotion_file(path):
    """Reads segmentation mot file with relational positioning
    Args:
        path (str): path to the ground truth egomotion file
    Returns:
        egomotion (ndarray): array in [n_frames, 4, 4] format
    """
    ego_file = open(path, "r")
    ego_lines = ego_file.readlines()
    transformations = np.zeros((len(ego_lines), 4, 4), dtype=np.float64)
    transformations[0, ...] = np.eye(4)
    for i in range(1, len(ego_lines)):
        prev_line = np.array(
            list(map(float, ego_lines[i - 1].split(" "))), dtype=np.float64
        )
        cur_line = np.array(list(map(float, ego_lines[i].split(" "))), dtype=np.float64)
        diff = cur_line - prev_line  # difference in angles and position
        rotation = np.deg2rad([diff[0], -diff[2], diff[1]])
        rotation = R.from_rotvec(rotation).as_matrix()
        translation = np.array([diff[3], -diff[5], diff[4]], dtype=np.float64)
        transformations[i, ...] = utils.rt2transformation(rotation, translation)
    ego_file.close()
    return transformations


def read_mots_vo_file(path):
    """Reads segmentation mot file with relational positioning
    Args:
        path (str): path to the ground truth egomotion file
    Returns:
        egomotion (ndarray): array in [n_frames, 4, 4] format
    """
    rot = np.load(path + "/rotations.npy", allow_pickle=True)
    trans = np.load(path + "/translations.npy", allow_pickle=True)
    transformations = np.zeros((rot.shape[0], 4, 4), dtype=np.float64)
    transformations[0, ...] = np.eye(4)
    for i in range(1, rot.shape[0]):
        t_diff = trans[i] - trans[i - 1]
        r_diff = (
            R.from_matrix(rot[i]).as_rotvec() - R.from_matrix(rot[i - 1]).as_rotvec()
        )
        r_diff = R.from_rotvec(r_diff).as_matrix()
        transformations[i, ...] = utils.rt2transformation(r_diff, t_diff)
    return transformations


# def read_motsynth_egomotion_file(path):
#     """Reads segmentation mot file with absolute positioning
#     Args:
#         path (str): path to the ground truth egomotion file
#     Returns:
#         egomotion (ndarray): array in [n_frames, 4, 4] format
#     """
#     ego_file = open(path, "r")
#     ego_lines = ego_file.readlines()
#     transformations = np.zeros((len(ego_lines), 4, 4), dtype=np.float64)
#     for i, line in enumerate(ego_lines):
#         line = list(map(float, line.split(" ")))
#         angles = np.array(line[:3], dtype=np.float64)
#         rotation = np.deg2rad([angles[0], -angles[2], angles[1]])
#         rotation = R.from_rotvec(rotation).as_matrix()
#         translation = np.array([line[3], -line[5], line[4]], dtype=np.float64)
#         transformations[i, ...] = utils.rt2transformation(rotation, translation)
#     ego_file.close()
#     return transformations
