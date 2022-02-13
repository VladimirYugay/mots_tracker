""" Splits motsynth dataset into train, validation and test"""
import logging
from pathlib import Path

import click
import cv2
import numpy as np
import open3d as o3d
from tqdm import tqdm

from mots_tracker import readers, vis_utils
from mots_tracker.io_utils import get_instance, load_yaml
from mots_tracker.readers.mot16_reader import DYNAMIC_SEQUENCES

_logger = logging.getLogger(__name__)


def get_gradient(depth):
    depth = depth.copy()
    sobelx = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=5)
    gradient = abs(sobelx + sobely)
    return gradient


def compute_egomotion(source_sample: dict, target_sample: dict) -> np.ndarray:
    """Compute egomotion between two consecutive frames

    Args:
        source_sample: dict with info about the frame
        target_sample: next dict with info about the frame
    Returns:
        np.ndarray: 4x4 transformation matrix
    """

    # verify_ptcd_order(source_sample)

    spanoptic, tpanoptic = (
        source_sample["panoptic_mask"],
        target_sample["panoptic_mask"],
    )
    source_depth, target_depth = source_sample["depth"], target_sample["depth"]
    source_img, target_img = source_sample["image"], target_sample["image"]

    # remove unstable depth regions
    source_gradient = get_gradient(source_depth)
    target_gradient = get_gradient(target_depth)
    grad_thres = 30

    source_depth[source_gradient > grad_thres] = 0
    source_img[source_gradient > grad_thres] = 0

    target_depth[target_gradient > grad_thres] = 0
    target_img[target_gradient > grad_thres] = 0

    # remove moving regions
    source_depth[spanoptic == 0] = 0
    source_img[spanoptic == 0] = 0

    target_depth[tpanoptic == 0] = 0
    target_img[tpanoptic == 0] = 0

    # remove the far depth
    depth_thres = 50

    source_img[source_depth > depth_thres] = 0
    source_depth[source_depth > depth_thres] = 0

    target_img[target_depth > depth_thres] = 0
    target_depth[target_depth > depth_thres] = 0

    print(source_img.shape, target_img.shape)
    print(source_depth.shape, target_depth.shape)
    vis_utils.plot_image(source_img)
    vis_utils.plot_image(target_img)
    # ICP
    height, width = source_depth.shape
    intrinsics = source_sample["intrinsics"]
    intrinsics = o3d.open3d.camera.PinholeCameraIntrinsic(
        width,
        height,
        fx=intrinsics[0][0],
        fy=intrinsics[1][1],
        cx=intrinsics[0][2],
        cy=intrinsics[1][2],
    )

    source_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(source_img),
        o3d.geometry.Image(source_depth),
        convert_rgb_to_intensity=True,
        depth_scale=1,
        depth_trunc=1e6,
    )

    target_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(target_img),
        o3d.geometry.Image(target_depth),
        convert_rgb_to_intensity=True,
        depth_scale=1,
        depth_trunc=1e6,
    )

    success, transformation, _ = o3d.odometry.compute_rgbd_odometry(
        source_rgbd, target_rgbd, intrinsics
    )

    return success, transformation


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
        transformations = []
        for frame_id in tqdm(range(reader.sequence_info[seq_id]["length"] - 1)):
            source_sample = reader.read_sample(seq_id, frame_id)
            target_sample = reader.read_sample(seq_id, frame_id + 1)
            suc, egomotion = compute_egomotion(source_sample, target_sample)
            np.set_printoptions(suppress=True, precision=3)
            print(suc)
            print(egomotion)
            break
            transformations.append(egomotion)
        break
        transformations = np.array(transformations)
        np.save(filename, transformations)


if __name__ == "__main__":
    main()
