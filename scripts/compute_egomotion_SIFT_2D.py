""" Splits motsynth dataset into train, validation and test"""
import logging
from pathlib import Path

import click
import cv2
import numpy as np

# import open3d as o3d
from tqdm import tqdm

from mots_tracker import readers, utils
from mots_tracker.io_utils import get_instance, load_yaml
from mots_tracker.readers.mot16_reader import DYNAMIC_SEQUENCES

_logger = logging.getLogger(__name__)


def preprocess_samples(source_sample, target_sample):
    height, width, _ = source_sample["image"].shape

    spanoptic, tpanoptic = (
        source_sample["panoptic_mask"],
        target_sample["panoptic_mask"],
    )

    # filter invalid classes and unknown flow values
    mask = np.ones((height, width), dtype=np.int8)
    mask[np.logical_or(spanoptic == 0, tpanoptic == 0)] = 0

    # filter too far depth regions
    max_depth = 100
    mask[
        np.logical_or(
            source_sample["depth"] > max_depth, target_sample["depth"] > max_depth
        )
    ] = 0

    # filter based on stable gradient
    # grad_threshold = 50  # since lower kills too much rigid objects around
    # sgradient = utils.compute_depth_gradient(source_sample["depth"])
    # tgradient = utils.compute_depth_gradient(target_sample["depth"])
    # mask[np.logical_or(sgradient > grad_threshold, tgradient > grad_threshold)] = 0
    return mask


def get_features(a, b):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(a, None)
    kp2, des2 = sift.detectAndCompute(b, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)
    matched_pixels = []
    for match in matches:
        img1_idx = match[0].queryIdx
        img2_idx = match[0].trainIdx
        (col1, row1) = kp1[img1_idx].pt
        (col2, row2) = kp2[img2_idx].pt
        matched_pixels.append([row1, col1, row2, col2])

    matchesMask = [[0, 0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matchesMask[i] = [1, 0]

    matched_pixels = []
    for a, b in matches:
        if a.distance >= 0.7 * b.distance:
            continue
        img1_idx = a.queryIdx
        img2_idx = a.trainIdx
        (col1, row1) = kp1[img1_idx].pt
        (col2, row2) = kp2[img2_idx].pt
        matched_pixels.append([row1, col1, row2, col2])
    return np.array(matched_pixels)


def compute_egomotion(source_sample: dict, target_sample: dict) -> np.ndarray:
    """Compute egomotion between two consecutive frames

    Args:
        source_sample: dict with info about the frame
        target_sample: next dict with info about the frame
    Returns:
        np.ndarray: 4x4 transformation matrix
    """

    height, width, _ = source_sample["image"].shape
    mask = preprocess_samples(source_sample, target_sample)

    simg, timg = source_sample["image"], target_sample["image"]
    matched_pixels = get_features(simg, timg).astype(np.int32)

    srows, scols = matched_pixels[:, 0], matched_pixels[:, 1]
    trows, tcols = matched_pixels[:, 2], matched_pixels[:, 3]
    valid_pixels = np.logical_and(mask[srows, scols] == 1, mask[trows, tcols] == 1)
    srows, scols = srows[valid_pixels], scols[valid_pixels]
    trows, tcols = trows[valid_pixels], tcols[valid_pixels]

    source_cloud = utils.rgbd2ptcloud(
        source_sample["image"], source_sample["depth"], source_sample["intrinsics"]
    )
    target_cloud = utils.rgbd2ptcloud(
        target_sample["image"], target_sample["depth"], target_sample["intrinsics"]
    )

    source_ids = srows * width + scols
    target_ids = trows * width + tcols

    source_cloud_pts = np.asarray(source_cloud.points)
    target_cloud_pts = np.asarray(target_cloud.points)
    source_cloud_pts = source_cloud_pts[source_ids]
    target_cloud_pts = target_cloud_pts[target_ids]

    T = utils.rigid_transform_3D(
        np.asmatrix(source_cloud_pts[:, [0, 2]]),
        np.asmatrix(target_cloud_pts[:, [0, 2]]),
        False,
    )
    transformation = np.eye(4)
    transformation[:3, :3] = np.array(
        [[T[0][0], 0, T[0][1]], [0, 1, 0], [T[1][0], 0, T[1][1]]]
    )
    transformation[0][3] = T[0][2]
    transformation[2][3] = T[1][2]

    return transformation


@click.command()
@click.option(
    "--cp",
    "--config_path",
    "config_path",
    default="configs/debug_config.yaml",
    type=click.Path(exists=True),
    help="Path to the dataset: MOTS, MOTSynth, KITTI",
)
@click.option(
    "--fap",
    "--floop_alignment_path",
    "floop_alignment_path",
    type=click.Path(exists=True),
    help="Output path of the for the split files",
)
@click.option(
    "--op",
    "--output_path",
    "output_path",
    default="data/output",
    type=click.Path(exists=True),
    help="Output path of the for the split files",
)
def main(config_path, floop_alignment_path, output_path):
    config = load_yaml(config_path)
    output_path = Path(output_path)
    floop_alignment_path = Path(floop_alignment_path)

    reader = get_instance(readers, "reader", config)
    for seq_id in config["reader"]["args"]["seq_ids"]:

        if not DYNAMIC_SEQUENCES[seq_id]:
            continue

        rotations = np.load(
            str(floop_alignment_path / "{}_floor_rotations.npy".format(seq_id))
        )
        translations = np.load(
            str(floop_alignment_path / "{}_floor_translations.npy".format(seq_id))
        )

        print("Processing", seq_id)
        filename = "{}_egomotion.npy".format(seq_id)
        filename = str(output_path / filename)
        transformations = []
        for frame_id in tqdm(range(0, reader.sequence_info[seq_id]["length"] - 1)):

            source_sample = reader.read_sample(seq_id, frame_id)
            target_sample = reader.read_sample(seq_id, frame_id + 15)

            source_sample["R_floor"] = rotations[frame_id]
            source_sample["t_floor"] = translations[frame_id]
            target_sample["R_floor"] = rotations[frame_id + 1]
            target_sample["t_floor"] = translations[frame_id + 1]

            egomotion = compute_egomotion(source_sample, target_sample)
            transformations.append(egomotion)
        transformations = np.array(transformations)
        np.save(filename, transformations)


if __name__ == "__main__":
    main()
