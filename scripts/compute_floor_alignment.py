""" Splits motsynth dataset into train, validation and test"""
import logging
from pathlib import Path

import click
import cv2
import numpy as np
from tqdm import tqdm

from mots_tracker import readers, utils
from mots_tracker.io_utils import get_instance, load_yaml
from mots_tracker.readers.mot16_reader import DYNAMIC_SEQUENCES

_logger = logging.getLogger(__name__)


def get_gradient(depth):
    depth = depth.copy()
    sobelx = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=5)
    gradient = abs(sobelx + sobely)
    return gradient


def get_floor_transform(cloud):
    plane_model, inliers = cloud.segment_plane(
        distance_threshold=0.01, ransac_n=3, num_iterations=1000
    )
    inlier_cloud = cloud.select_down_sample(inliers)
    a, b, c, d = plane_model
    plane_normal = np.array(plane_model[:3])
    xz_normal = np.array([0, 1, 0])
    rotation_angle = utils.compute_angle(xz_normal, plane_normal)
    # define in which direction to rotate the plane
    if c > 0:  # plane looks up
        rotation_angle = -rotation_angle
    R = cloud.get_rotation_matrix_from_xyz(np.array([rotation_angle, 0, 0]))
    inlier_cloud.rotate(R)
    shift_num = np.asanyarray(inlier_cloud.points)[:, 1].min()
    t = np.array([0, -shift_num, 0])
    return R, t


def compute_floor_transform(sample: dict) -> tuple:
    """Compute egomotion between two consecutive frames

    Args:
        sample: dict with info about the frame
    Returns:
        tuple of rotation matrix and translation vector
    """
    img, depth, panoptic = (
        sample["image"].copy(),
        sample["depth"].copy(),
        sample["panoptic_mask"].copy(),
    )

    mask = np.ones_like(panoptic)
    mask[panoptic == 0] = 0
    mask[depth > 50] = 0
    mask[get_gradient(depth) > 30] = 0

    img[mask == 0] = 0
    depth[mask == 0] = 0

    cloud = utils.rgbd2ptcloud(img, depth, sample["intrinsics"])
    R, t = get_floor_transform(cloud)
    return R, t


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
    "--op",
    "--output_path",
    "output_path",
    default="data/output",
    type=click.Path(exists=True),
    help="Output path of the for the split files",
)
def main(config_path, output_path):
    config = load_yaml(config_path)
    output_path = Path(output_path)
    reader = get_instance(readers, "reader", config)
    for seq_id in config["reader"]["args"]["seq_ids"]:

        if not DYNAMIC_SEQUENCES[seq_id]:
            continue

        print("Processing", seq_id)
        filename = "{}_egomotion.npy".format(seq_id)
        filename = str(output_path / filename)
        rotaitons, translations = [], []
        for frame_id in tqdm(range(350, reader.sequence_info[seq_id]["length"])):
            sample = reader.read_sample(seq_id, frame_id)
            R, t = compute_floor_transform(sample)

            cloud = utils.rgbd2ptcloud(
                sample["image"], sample["depth"], sample["intrinsics"]
            )
            from copy import deepcopy

            cloud_rot = deepcopy(cloud)
            cloud_rot.paint_uniform_color([0, 1, 0])

            rotaitons.append(R)
            translations.append(t)

            cloud_rot.rotate(R)
            cloud_rot.translate(t)

            from mots_tracker import vis_utils

            vis_utils.plot_ptcloud([cloud_rot, cloud], True)
            break
        break

        rotaitons = np.array(rotaitons)
        translations = np.array(translations)
        np.save("{}_floor_rotations.npy".format(seq_id), rotaitons)
        np.save("{}_floor_translations.npy".format(seq_id), translations)


if __name__ == "__main__":
    main()
