""" Splits motsynth dataset into train, validation and test"""
from multiprocessing import Pool
from pathlib import Path

import click
import cv2
import numpy as np
from tqdm import tqdm

from mots_tracker import readers, utils
from mots_tracker.io_utils import get_instance, load_yaml


def multi_run_wrapper(args):
    """ Unpacks argument for running on multiple cores """
    return compute_floor_alignment_parallel(*args)


def compute_floor_alignment_parallel(reader, seq_id, output_path):

    rotaitons, translations = [], []
    for frame_id in range(reader.sequence_info[seq_id]["length"]):
        print("Sequence: {}, frame: {} out of {}".format(
            seq_id, frame_id, reader.sequence_info[seq_id]["length"]
        ))
        sample = reader.read_sample(seq_id, frame_id)
        R, t = compute_floor_transform(sample)
        rotaitons.append(R)
        translations.append(t)

    rotaitons = np.array(rotaitons)
    translations = np.array(translations)
    np.save(str(output_path / "{}_floor_rotations.npy".format(seq_id)), rotaitons)
    np.save(
        str(output_path / "{}_floor_translations.npy".format(seq_id)), translations
    )


def get_floor_transform(cloud):
    plane_model, inliers = cloud.segment_plane(
        distance_threshold=0.01, ransac_n=3, num_iterations=1000
    )
    inlier_cloud = cloud.select_down_sample(inliers)
    a, b, c, d = plane_model
    if inlier_cloud.is_empty():
        return np.eye(3), np.zeros(3)
    plane_normal = np.array(plane_model[:3])
    xz_normal = np.array([0, 1, 0])
    rotation_angle = utils.compute_angle(xz_normal, plane_normal)
    if c > 0:  # plane looks up
        rotation_angle = -rotation_angle    
    R = cloud.get_rotation_matrix_from_xyz(np.array([rotation_angle, 0, 0]))
    inlier_cloud.rotate(R)
    shift_num = np.asanyarray(inlier_cloud.points)[:, 1].mean()
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
    mask[utils.compute_depth_gradient(depth) > 30] = 0

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
    pool = Pool(4)
    # print("Running on {} cores".format(config.get("cores", 4)))
    args = [
        (reader, seq_id, output_path) 
        for seq_id in config["reader"]["args"]["seq_ids"]
    ]    
    pool.map(multi_run_wrapper, args)


if __name__ == "__main__":
    main()
