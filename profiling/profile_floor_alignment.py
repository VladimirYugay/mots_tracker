""" profiling MOTS dataset """
from copy import deepcopy

import numpy as np

from mots_tracker import readers, utils, vis_utils
from mots_tracker.io_utils import get_instance, load_yaml

np.set_printoptions(suppress=True, precision=3)


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

    # original - original color
    orig = deepcopy(inlier_cloud)
    # rotated - Green
    orig_rot = deepcopy(inlier_cloud)
    orig_rot.paint_uniform_color([0, 1, 0])
    # rotated and translated - Blue
    orig_rot_trans = deepcopy(inlier_cloud)
    orig_rot_trans.paint_uniform_color([0, 0, 1])

    orig_rot.rotate(R, center=True)

    shift_num = np.asanyarray(orig_rot.points)[:, 1].mean()
    orig_rot_trans.rotate(R, center=True)
    orig_rot_trans.translate(np.array([0, -shift_num, 0]))

    print(np.asanyarray(orig_rot.points)[:, 1].mean())
    print(np.asanyarray(orig_rot_trans.points)[:, 1].mean())

    vis_utils.plot_ptcloud([orig, orig_rot, orig_rot_trans], True)


def profile_alignment(sample: dict) -> tuple:
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
    get_floor_transform(cloud)


def main():
    config_path = "./configs/debug_config.yaml"
    config = load_yaml(config_path)
    reader = get_instance(readers, "reader", config)
    seq_id, frame_id = "MOT16-14", 0  # 300 for challenge
    source_sample = reader.read_sample(
        seq_id,
        frame_id,
    )
    profile_alignment(source_sample)


if __name__ == "__main__":
    main()
