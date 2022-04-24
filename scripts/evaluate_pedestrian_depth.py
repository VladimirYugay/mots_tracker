from ctypes import util
import json
import pickle
from argparse import Namespace
from pathlib import Path
import itertools
import cv2
import numpy as np
from tqdm import tqdm
from src.mots_tracker import utils
from src.mots_tracker.readers.reader_helpers import read_mot_seg_file


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths"""
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def load_motsynth_depth_image(img_path):
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


def read_panoptic_img(panoptic_path, color2coco, 
    classes2category, categories, include_cats=[], exclude_cats=[]):
    """Reads depth map file
    Args:
        seq_id: sequence ud
        frame_id: frame id
    Returns:
        np.ndarray: float depth map
    """
    img = utils.load_image(panoptic_path)
    img = np.array(img)
    colors = np.unique(img.reshape(-1, img.shape[2]), axis=0)
    panoptic_img = np.zeros_like(img)
    panoptic_mask = np.zeros((img.shape[0], img.shape[1]))
    for color in colors:
        color_str = " ".join(str(e) for e in color)
        coco_class = color2coco.get(color_str, None)
        if coco_class is None:
            continue
        category = classes2category[coco_class]
        if include_cats and category not in include_cats:
            continue
        if exclude_cats and category in exclude_cats:
            continue
        final_color = category_colors[category]
        color_indices = np.where(np.all(img == color, axis=-1))
        panoptic_img[color_indices] = final_color
        panoptic_mask[color_indices] = categories.index(category) + 1
    return panoptic_img, panoptic_mask


def _init_color2coco(meta_data):
    """ Initializes mapping of color to coco """
    color2coco = {}
    for stuff_class, stuff_color in zip(
        meta_data.stuff_classes, meta_data.stuff_colors):
        color = " ".join(str(e) for e in stuff_color)
        color2coco[color] = stuff_class
    for thing_class, thing_color in zip(
        meta_data.thing_classes, meta_data.thing_colors):
        color = " ".join(str(e) for e in thing_color)
        color2coco[color] = thing_class
    return color2coco


category_colors = {
    "sky": np.array([70, 130, 180]),
    "pedestrian": np.array([255, 0, 0]),
    "other": np.array([255, 215, 0]),
    "occluder_moving": np.array([0, 100, 100]),
    "occluder_static": np.array([0, 0, 230]),
    "building": np.array([0, 255, 0]),
    "road": np.array([50, 50, 50]),
    "pavement": np.array([100, 100, 100]),
    "ground": np.array([10, 200, 10]),
}

step = 1000000
palette = list(itertools.product(np.arange(1, 256), repeat=3))
color_ids = [palette[i::step] for i in range(step)]

json_path = "/storage/local/yugay/projects/mots_tracker/category2classes.json"
metadata_path = "/usr/stud/yugay/MOT16_helper_code/metaDataClasses.pkl"
with open(json_path) as json_file:
    classes2category = json.load(json_file)
with open(metadata_path, "rb") as pkl_file:
    meta_data = Namespace(**pickle.load(pkl_file))
color2coco = _init_color2coco(meta_data)
categories = list(category_colors.keys())

seq_file_path = "/storage/local/yugay/datasets/MOTSynth_annotations/official_split/test.txt"
with open(seq_file_path, 'r') as file:
    seq_ids = set([line.strip() for line in file.readlines()])    


panoptic_path = Path("/storage/user/dendorfp/MOTSynth/panoptic")
panoptic_paths = sorted([p for p in panoptic_path.glob('*') if str(p.parts[-1]) in seq_ids])

# gt_depth_new
gt_depth_path = Path("/storage/local/yugay/datasets/MOTSynth_annotations/all/")
gt_depth_paths = sorted([p for p in gt_depth_path.glob('*') if str(p.parts[-1]) in seq_ids])

# best_motsynth.pt
est_depth_path = Path("/usr/stud/yugay/MOT16/MOTSynth_depth/")
est_depth_paths = sorted([p for p in est_depth_path.glob('*') if str(p.parts[-1]) in seq_ids])


import matplotlib.pyplot as plt 
MIN_DEPTH = 0.001


ped_err_file = open('pedestrian_depth_error.txt', 'w')
ground_err_file = open('ground_depth_error.txt', 'w')
ped_errors, ground_errors = [], []


for seq_id in tqdm(range(len(gt_depth_paths))):
    gt_files = sorted((gt_depth_paths[seq_id] / 'gt_depth_new').glob('*'))
    est_files = sorted((est_depth_paths[seq_id] / 'best_motsynth.pt').glob('*'))
    panoptic_files = sorted((panoptic_paths[seq_id]).glob('*'))
    for i in tqdm(range(0, len(gt_files), 36), leave=False):
        gt_depth = load_motsynth_depth_image(str(gt_files[i]))
        gt_depth = np.clip(gt_depth, MIN_DEPTH, 100)

        est_depth = np.load(str(est_files[i]))["arr_0"]
        est_depth = utils.interpolate_depth(est_depth, (1920, 1080))
        est_depth = np.clip(est_depth, MIN_DEPTH, 100)

        _, ground_mask = read_panoptic_img(
            panoptic_files[i], color2coco, classes2category, categories,
            ["road", "pavement", "ground"])

        _, pedestrian_mask = read_panoptic_img(
            panoptic_files[i], color2coco, classes2category, categories, 
            ["pedestrian"])            

        ped_gt_depth = gt_depth[pedestrian_mask != 0].flatten()
        ped_est_depth = est_depth[pedestrian_mask != 0].flatten()
        
        perr = compute_errors(ped_gt_depth, ped_est_depth)
        ped_errors.append(perr)
        print(*perr, file=ped_err_file)

        ground_gt_depth = gt_depth[ground_mask != 0].flatten()
        ground_est_depth = est_depth[ground_mask != 0].flatten()
        
        gerr = compute_errors(ground_gt_depth, ground_est_depth)
        ground_errors.append(gerr)
        print(*gerr, file=ground_err_file)

ped_errors = np.array(ped_errors)
ground_errors = np.array(ground_errors)
print("Pedestrian error", np.mean(ped_errors, axis=0))
print("Ground error", np.mean(ground_errors, axis=0))