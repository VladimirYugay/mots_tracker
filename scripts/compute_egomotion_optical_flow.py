""" Splits motsynth dataset into train, validation and test"""
import logging
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np

from mots_tracker import readers, utils
from mots_tracker.io_utils import get_instance, load_yaml
from mots_tracker.readers.mot16_reader import DYNAMIC_SEQUENCES

_logger = logging.getLogger(__name__)


def vis_optical_flow_distribution(horizontal, vertical):
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.hist(horizontal, bins=50)
    ax1.set_title("Horizontal flow distribution")
    ax2.hist(vertical, bins=50)
    ax2.set_title("Vertical flow distribution")
    plt.show()


def vis_correspondences(img_left, img_right, source_pxls, target_pxls):
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.imshow(img_left)
    ax1.scatter(source_pxls[1], source_pxls[0], marker=",")
    ax1.set_title("Left image")
    ax2.imshow(img_right)
    ax2.scatter(target_pxls[1], target_pxls[0], marker=",")
    ax2.set_title("Right image")
    plt.show()


def vis_movement(img, scol, srow, tcol, trow):
    plt.imshow(img)
    plt.scatter(scol, srow, marker=",", color="blue")
    plt.scatter(tcol, trow, marker="x", color="red")
    for sy, sx, ty, tx in zip(srow, scol, trow, tcol):
        plt.plot([sx, tx], [sy, ty], color="k")
    plt.show()


def get_topk_displacement_ids(horizontal, vertical, k=100):
    height, width = horizontal.shape
    total_flow = horizontal + vertical
    kth_largest = np.sort(total_flow.flatten())[-k]
    return np.where(total_flow >= kth_largest)


def get_pt_cloud_from_pixels(sample, rows, cols):
    empty_img = np.zeros_like(sample["image"])
    empty_depth = np.zeros_like(sample["depth"])
    empty_img[rows, cols] = sample["image"][rows, cols]
    empty_depth[rows, cols] = sample["depth"][rows, cols]
    cloud = utils.rgbd2ptcloud(empty_img, empty_depth, sample["intrinsics"])
    return np.asmatrix(cloud.points)


def prune_clouds(clouda, cloudb):
    num_a, _ = clouda.shape
    num_b, _ = cloudb.shape
    n = min(num_a, num_b)
    return clouda[:n, :], cloudb[:n, :]


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


def compute_egomotion(source_sample: dict, target_sample: dict) -> np.ndarray:
    """Compute egomotion between two consecutive frames

    Args:
        source_sample: dict with info about the frame
        target_sample: next dict with info about the frame
    Returns:
        np.ndarray: 4x4 transformation matrix
    """
    height, width, _ = source_sample["image"].shape
    flow = source_sample["optical_flow"]
    horizontal, vertical = flow[..., 0], flow[..., 1]
    panoptic = source_sample["panoptic_mask"]
    # exclude dynamic objects optical flow pixels and unknown flow values

    horizontal[panoptic == 0] = horizontal.min()
    horizontal[horizontal == 1e9] = horizontal.min()
    vertical[panoptic == 0] = vertical.min()
    vertical[vertical == 1e9] = vertical.min()
    # top dynamic flow pixels
    k = int(1e5)
    k = int(1e3)
    rows, cols = get_topk_displacement_ids(horizontal, vertical, k)
    vdisp, hdisp = vertical[rows, cols], horizontal[rows, cols]
    scols, srows = (cols.copy(), rows.copy())
    tcols, trows = (cols.copy() + np.rint(hdisp), rows.copy() + np.rint(vdisp))
    tcols, trows = tcols.astype(np.int), trows.astype(np.int)
    print("Before filtering", scols.shape, srows.shape, tcols.shape, trows.shape)
    valid_pixels = (tcols < width) & (trows < height)
    scols, srows = scols[valid_pixels], srows[valid_pixels]
    tcols, trows = tcols[valid_pixels], trows[valid_pixels]
    print("After filtering", scols.shape, srows.shape, tcols.shape, trows.shape)

    # vis_movement(source_sample["image"], scols, srows, tcols, trows)

    # vis_correspondences(
    #     source_sample["image"],
    #     target_sample["image"],
    #     (scols, srows),
    #     (tcols, trows))

    source_cloud = get_pt_cloud_from_pixels(source_sample, srows, scols)
    target_cloud = get_pt_cloud_from_pixels(target_sample, trows, tcols)
    source_cloud, target_cloud = prune_clouds(source_cloud, target_cloud)

    print(source_cloud.shape, target_cloud.shape)
    transformation = rigid_transform_3D(source_cloud, target_cloud, False)
    print(transformation)


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
        transformations = []
        for frame_id in range(reader.sequence_info[seq_id]["length"] - 1):
            source_sample = reader.read_sample(seq_id, frame_id)
            target_sample = reader.read_sample(seq_id, frame_id)
            egomotion = compute_egomotion(source_sample, target_sample)
            transformations.append(egomotion)
        transformations = np.array(transformations)
        np.save(filename, transformations)


if __name__ == "__main__":
    main()
