""" Splits motsynth dataset into train, validation and test"""
import logging

import click
import numpy as np

from mots_tracker import readers, utils
from mots_tracker.io_utils import get_instance, load_yaml

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

    empty_img = np.zeros_like(sample_left["image"])
    empty_depth = np.zeros_like(sample_left["depth"])

    left_img, left_depth = empty_img.copy(), empty_depth.copy()
    left_img[kpts_left[:, 1], kpts_right[:, 0]] = sample_left["image"][
        kpts_left[:, 1], kpts_left[:, 0]
    ]
    left_depth[kpts_left[:, 1], kpts_right[:, 0]] = sample_left["depth"][
        kpts_left[:, 1], kpts_left[:, 0]
    ]

    right_img, right_depth = empty_img.copy(), empty_depth.copy()
    right_img[kpts_right[:, 1], kpts_right[:, 0]] = sample_right["image"][
        kpts_right[:, 1], kpts_right[:, 0]
    ]
    right_depth[kpts_left[:, 1], kpts_right[:, 0]] = sample_right["depth"][
        kpts_right[:, 1], kpts_right[:, 0]
    ]

    cloud_left = utils.rgbd2ptcloud(left_img, left_depth, sample_left["intrinsics"])
    cloud_right = utils.rgbd2ptcloud(right_img, right_depth, sample_right["intrinsics"])

    cloud_left_pts = np.asmatrix(cloud_left.points)
    cloud_right_pts = np.asmatrix(cloud_right.points)

    print(cloud_left_pts.shape)
    print(cloud_right_pts.shape)
    t = rigid_transform_3D(cloud_left_pts, cloud_right_pts, None)
    np.set_printoptions(suppress=True, precision=3)
    print(t)


def rigid_transform_3D(A, B, scale):
    assert A.shape == B.shape
    N = A.shape[0]  # total points
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))
    # dot is matrix multiplication for array
    if scale:
        H = np.transpose(BB) * AA / N
    else:
        # print(BB.T.shape, AA.shape)
        # H = np.matmul(BB.T, AA)
        H = np.transpose(BB) * AA
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T * U.T
    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T * U.T
    if scale:
        varA = np.var(A, axis=0).sum()
        c = 1 / (1 / varA * np.sum(S))  # scale factor
        t = -R * (centroid_B.T * c) + centroid_A.T
    else:
        c = 1
        t = -R * centroid_B.T + centroid_A.T
    R = R * c
    transformation = np.zeros((4, 4))
    transformation[3, 3] = 1
    transformation[:3, :3] = R
    transformation[:3, 3] = t.T
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
    for seq_id in config["reader"]["args"]["seq_ids"]:
        for frame_id in range(reader.sequence_info[seq_id]["length"] - 1):
            sample_left = reader.read_sample(seq_id, frame_id)
            sample_right = reader.read_sample(seq_id, frame_id)
            egomotion = compute_egomotion(sample_left, sample_right)
            print(egomotion)
            break
        break


if __name__ == "__main__":
    main()
