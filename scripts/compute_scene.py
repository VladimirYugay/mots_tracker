""" Splits motsynth dataset into train, validation and test"""
import copy
import logging
from pathlib import Path

import click
import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from tqdm import tqdm

from mots_tracker import readers, utils, vis_utils
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


def rigid_transform_3D(A, B, scale):
    assert len(A) == len(B)

    N = A.shape[0]  # total points

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # center the points
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))

    # dot is matrix multiplication for array
    if scale:
        H = np.transpose(BB) * AA / N
    else:
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


def verify_ptcd_order(sample):
    # cloud = utils.rgbd2ptcloud(
    #     sample["image"], sample["depth"], sample["intrinsics"])
    height, width, _ = sample["image"].shape
    img = sample["image"].copy()
    img[:5, ...] = [0, 255, 0]
    # vis_utils.plot_image(img)
    cloud = utils.rgbd2ptcloud(img, sample["depth"], sample["intrinsics"])
    # vis_utils.plot_ptcloud(cloud)
    colors = np.asarray(cloud.colors)
    assert np.all(
        colors[
            : 5 * width,
        ]
        == [0.0, 1.0, 0.0]
    )

    img = sample["image"].copy()
    img[100:150, ...] = [0, 255, 0]
    # vis_utils.plot_image(img)
    cloud = utils.rgbd2ptcloud(img, sample["depth"], sample["intrinsics"])
    # vis_utils.plot_ptcloud(cloud)
    colors = np.asarray(cloud.colors)
    assert np.all(
        colors[
            100 * width : 100 * width + 50,
        ]
        == [0.0, 1.0, 0.0]
    )

    img = sample["image"].copy()
    img[100:150, 350:400, :] = [0, 255, 0]
    # vis_utils.plot_image(img)
    cloud = utils.rgbd2ptcloud(img, sample["depth"], sample["intrinsics"])
    # vis_utils.plot_ptcloud(cloud)
    colors = np.asarray(cloud.colors)
    assert np.all(
        colors[
            width * 100 + 350 : width * 100 + 400,
        ]
        == [0.0, 1.0, 0.0]
    )


def vis_points(pts):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pts)
    vis_utils.plot_ptcloud(cloud)


def get_gradient(depth, panoptic):
    depth = depth.copy()
    depth[panoptic == 0] = 0
    sobelx = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=5)
    gradient = abs(sobelx + sobely)
    return gradient


def compute_egomotion(sample: dict, target_sample: dict) -> np.ndarray:
    """Compute egomotion between two consecutive frames

    Args:
        sample: dict with info about the frame
        target_sample: next dict with info about the frame
    Returns:
        np.ndarray: 4x4 transformation matrix
    """

    # verify_ptcd_order(sample)

    height, width, _ = sample["image"].shape
    flow = sample["optical_flow"]
    hflow, vflow = flow[..., 0], flow[..., 1]

    spanoptic, tpanoptic = (
        sample["panoptic_mask"],
        target_sample["panoptic_mask"],
    )

    # filter invalid classes and unknown flow values
    mask = np.ones((height, width), dtype=np.int8)
    mask[np.logical_or(spanoptic == 0, tpanoptic == 0)] = 0
    mask[np.logical_or(hflow == 1e9, vflow == 1e9)] = 0

    # filter too far depth regions
    max_depth = 50
    mask[
        np.logical_or(sample["depth"] > max_depth, target_sample["depth"] > max_depth)
    ] = 0

    # filter based on stable gradient
    grad_threshold = 8
    sgradient = get_gradient(sample["depth"], spanoptic)
    tgradient = get_gradient(target_sample["depth"], tpanoptic)
    mask[np.logical_or(sgradient > grad_threshold, tgradient > grad_threshold)] = 0

    # get the ids of interest
    srows, scols = np.where(mask != 0)
    hdisp, vdisp = hflow[srows, scols], vflow[srows, scols]
    hdisp, vdisp = np.rint(hdisp).astype(np.int32), np.rint(vdisp).astype(np.int32)
    trows, tcols = srows + vdisp, scols + hdisp

    # remove the out of boundaries pixels
    valid_pixels = (tcols < width) & (trows < height)
    srows, scols = srows[valid_pixels], scols[valid_pixels]
    trows, tcols = trows[valid_pixels], tcols[valid_pixels]

    # we want to preserve the semantic class between the pixels
    valid_cls = spanoptic[srows, scols] == tpanoptic[trows, tcols]
    srows, scols = srows[valid_cls], scols[valid_cls]
    trows, tcols = trows[valid_cls], tcols[valid_cls]

    cloud = utils.rgbd2ptcloud(sample["image"], sample["depth"], sample["intrinsics"])
    target_cloud = utils.rgbd2ptcloud(
        target_sample["image"], target_sample["depth"], target_sample["intrinsics"]
    )
    source_clrs = np.asanyarray(cloud.colors)
    target_clrs = np.asarray(target_cloud.colors)
    cloud_pts = np.asarray(cloud.points)
    target_cloud_pts = np.asarray(target_cloud.points)

    source_ids = srows * width + scols
    target_ids = trows * width + tcols
    source_pts = cloud_pts[source_ids]
    target_pts = target_cloud_pts[target_ids]

    source_pts = np.asmatrix(source_pts)
    target_pts = np.asmatrix(target_pts)
    transformation = rigid_transform_3D(source_pts, target_pts, False)

    new_cloud = o3d.geometry.PointCloud()
    new_cloud.points = o3d.utility.Vector3dVector(cloud_pts[source_ids])
    new_cloud.colors = o3d.utility.Vector3dVector(source_clrs[source_ids])

    new_target_cloud = o3d.geometry.PointCloud()
    new_target_cloud.points = o3d.utility.Vector3dVector(target_cloud_pts[target_ids])
    new_target_cloud.colors = o3d.utility.Vector3dVector(target_clrs[target_ids])

    return transformation, new_cloud, new_target_cloud


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries(
        [source_temp, target_temp],
        zoom=0.4459,
        front=[0.9288, -0.2951, -0.2242],
        lookat=[1.6784, 2.0612, 1.4451],
        up=[-0.3402, -0.9189, -0.1996],
    )


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
        # egomotion = np.load(
        #     "/home/vy/university/thesis/datasets/MOT16\
        #         /egomotion_optical_flow/MOT16-14_egomotion.npy"
        # )
        egomotion = np.load("/usr/stud/yugay/MOT16/egomotion_optical_flow/MOT16-14_egomotion.npy")
        transformations = []
        T = np.eye(4)
        clouds = []
        for frame_id in tqdm(range(reader.sequence_info[seq_id]["length"] - 1)):

            if frame_id % 30 == 0:
                sample = reader.read_sample(seq_id, frame_id)
                panoptic = sample["panoptic_mask"]
                static_img = sample["image"]
                static_depth = sample["depth"]
                height, width = panoptic.shape

                mask = np.ones((height, width), dtype=np.int8)
                mask[panoptic == 0] = 0
                max_depth = 30
                mask[static_depth > max_depth] = 0
                grad_threshold = 30
                gradient = utils.compute_depth_gradient(sample["depth"])
                mask[gradient > grad_threshold] = 0

                static_img[mask == 0] = 0
                static_depth[mask == 0] = 0

                cloud = utils.rgbd2ptcloud(
                    static_img, static_depth, sample["intrinsics"]
                )
                cloud.transform(T)

                if not clouds:
                    clouds = cloud
                else:
                    clouds += cloud
            T = T.dot(np.linalg.inv(egomotion[frame_id]))

            # break
            # transformations.append(egomotion)
        o3d.io.write_point_cloud('cloud.pcd', clouds)
        # vis_utils.plot_ptcloud(clouds, show_frame=True)
        break
        transformations = np.array(transformations)
        np.save(filename, transformations)


if __name__ == "__main__":
    main()
