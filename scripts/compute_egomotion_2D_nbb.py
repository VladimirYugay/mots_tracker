""" Splits motsynth dataset into train, validation and test"""
import logging
from pathlib import Path

import click
import numpy as np

from mots_tracker import readers, utils
from mots_tracker.io_utils import get_instance, load_yaml
from mots_tracker.readers.mot16_reader import DYNAMIC_SEQUENCES

_logger = logging.getLogger(__name__)


def compute_egomotion(sample_left: dict, sample_right: dict) -> np.ndarray:
    """Compute egomotion between two consecutive frames

    Args:
        sample_left: dict with info about the frame
        sample_right: next dict with info about the frame
    Returns:
        np.ndarray: 4x4 transformation matrix
    """
    kpts_left = sample_left["current_correspondence"]
    kpts_right = sample_right["next_correspondence"]
    lcol, lrow = kpts_left[:, 0], kpts_left[:, 1]
    rcol, rrow = kpts_right[:, 0], kpts_right[:, 1]
    panoptic_left = sample_left["panoptic_mask"]
    panoptic_right = sample_left["panoptic_mask"]

    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    ax1.imshow(sample_left["image"])
    ax1.scatter(kpts_left[:, 0], kpts_left[:, 1], c=np.arange(kpts_left.shape[0]))
    ax2.imshow(sample_right["image"])
    ax2.scatter(kpts_right[:, 0], kpts_right[:, 1], c=np.arange(kpts_left.shape[0]))
    plt.show()
    # return

    empty_img = np.zeros_like(sample_left["image"])
    empty_depth = np.zeros_like(sample_left["depth"])

    left_img, left_depth = empty_img.copy(), empty_depth.copy()
    left_img[lrow, lcol] = sample_left["image"][lrow, lcol]
    left_depth[lrow, lcol] = sample_left["depth"][lrow, lcol]
    left_img[panoptic_left == 0] = 0
    left_depth[panoptic_left == 0] = 0
    left_img[left_depth > 50] = 0
    left_depth[left_depth > 50] = 0
    left_img[utils.compute_depth_gradient(left_depth) > 30] = 0
    left_depth[utils.compute_depth_gradient(left_depth) > 30] = 0

    right_img, right_depth = empty_img.copy(), empty_depth.copy()
    right_img[rrow, rcol] = sample_right["image"][rrow, rcol]
    right_depth[rrow, rcol] = sample_right["depth"][rrow, rcol]
    right_img[panoptic_right == 0] = 0
    right_depth[panoptic_right == 0] = 0
    right_img[right_depth > 50] = 0
    right_depth[right_depth > 50] = 0
    right_img[utils.compute_depth_gradient(right_depth) > 30] = 0
    right_depth[utils.compute_depth_gradient(right_depth) > 30] = 0

    cloud_left = utils.rgbd2ptcloud(left_img, left_depth, sample_left["intrinsics"])
    cloud_right = utils.rgbd2ptcloud(right_img, right_depth, sample_right["intrinsics"])

    from mots_tracker import vis_utils

    cloud_right.paint_uniform_color([0, 1, 0])
    vis_utils.plot_ptcloud([cloud_left, cloud_right], True)
    return

    cloud_left_pts = np.asarray(cloud_left.points)[:, [0, 2]]
    cloud_right_pts = np.asarray(cloud_right.points)[:, [0, 2]]

    T = utils.rigid_transform_3D(cloud_left_pts, cloud_right_pts, None)
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
    "--op",
    "--output_path",
    "output_path",
    default="data/output",
    type=click.Path(exists=True),
    help="Output path of the for the split files",
)
def main(config_path, output_path):
    config = load_yaml(config_path)
    reader = get_instance(readers, "reader", config)
    output_path = Path(output_path)
    for seq_id in config["reader"]["args"]["seq_ids"]:
        if not DYNAMIC_SEQUENCES[seq_id]:
            continue
        print("Processing", seq_id)
        filename = "{}_egomotion.npy".format(seq_id)
        filename = str(output_path / filename)
        transformations = []
        for frame_id in range(reader.sequence_info[seq_id]["length"] - 1):
            sample_left = reader.read_sample(seq_id, frame_id)
            sample_right = reader.read_sample(seq_id, frame_id + 1)
            egomotion = compute_egomotion(sample_left, sample_right)
            transformations.append(egomotion)
        transformations = np.array(transformations)
        np.save(filename, transformations)


if __name__ == "__main__":
    main()
