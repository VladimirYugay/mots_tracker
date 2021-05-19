""" profiling gt bb generation from motsynth """
import numpy as np

from mots_tracker import utils, vis_utils
from mots_tracker.readers import MOTSynthReader


def profile_absolute_egomotion_trajectory(reader, seq_id, start=0, stop=10):
    """See the egomotion trajectory
    NOTE: works only in case of absolute positioning
    """
    np.set_printoptions(suppress=True, precision=3)
    p_0 = np.array([0, 0, 0, 1])
    poses = []
    for frame_id in range(start, stop):
        sample = reader.read_sample(seq_id, frame_id)
        p_acc = sample["egomotion"].dot(p_0)
        poses.append(p_acc.T)
    poses = np.array(poses)
    np.save("trajectory.npy", poses)
    vis_utils.plot_3d_pts(poses, True)
    vis_utils.plot_2d_pts(poses[:, [0, 2]], axis_names=("X", "Z"))
    vis_utils.plot_2d_pts(poses[:, [0, 1]], axis_names=("X", "Y"))
    vis_utils.plot_2d_pts(poses[:, [1, 2]], axis_names=("Y", "Z"))


def profile_relative_egomotion_trajectory(reader, seq_id, start=0, stop=10):
    """See the egomotion trajectory
    NOTE: works only in case of relative egomotion
    """
    np.set_printoptions(suppress=True, precision=5)
    T_acc = np.identity(4)
    p_acc = np.array([0.0, 0.0, 0.0, 1.0])
    p_0 = np.array([0.0, 0.0, 0.0, 1.0])
    poses = p_acc
    for frame_id in range(start + 1, stop):
        T = reader.read_sample(seq_id, frame_id)["egomotion"]
        T_acc = T_acc.dot(T)
        p_acc = T_acc.dot(p_0)
        poses = np.vstack((poses, p_acc))
    vis_utils.plot_3d_pts(poses, True)


def profile_absolute_egomotion_transform(reader, seq_id="045", frame_id=348):
    """ See the egomotion trajectory applied to clouds"""
    np.set_printoptions(suppress=True, precision=3)
    sample_left = reader.read_sample(seq_id, frame_id)
    sample_right = reader.read_sample(seq_id, frame_id + 1)

    scene_left = utils.rgbd2ptcloud(
        sample_left["image"], sample_left["depth"], sample_left["intrinsics"]
    )
    scene_right = utils.rgbd2ptcloud(
        sample_right["image"], sample_right["depth"], sample_right["intrinsics"]
    )
    scene_right.paint_uniform_color([0, 1, 0])
    vis_utils.plot_ptcloud([scene_left, scene_right], False)
    T_left, T_right = sample_left["egomotion"], sample_right["egomotion"]
    scene_left.transform(T_left)
    scene_right.transform(T_right)
    vis_utils.plot_ptcloud([scene_left, scene_right], False)


def profile_relative_egomotion_transform(reader, seq_id="045", frame_id=0):
    """ See the egomotion trajectory applied to clouds"""
    np.set_printoptions(suppress=True, precision=3)
    sample_left = reader.read_sample(seq_id, frame_id)
    sample_right = reader.read_sample(seq_id, frame_id + 1)

    T_right = sample_right["egomotion"]

    scene_left = utils.rgbd2ptcloud(
        sample_left["image"], sample_left["depth"], sample_left["intrinsics"]
    )
    scene_left.paint_uniform_color([0, 1, 0])
    scene_right = utils.rgbd2ptcloud(
        sample_right["image"], sample_right["depth"], sample_right["intrinsics"]
    )

    vis_utils.plot_ptcloud([scene_left, scene_right], False)
    scene_right.transform(T_right)
    vis_utils.plot_ptcloud([scene_left, scene_right], False)


def main():
    """ visual profiling for generated motsynth bb """
    config = {
        "depth_path": "gt_depth",
        "egomotion_path": "egomotion",
        "read_masks": True,
        "read_boxes": True,
        "gt_path": "/home/vy/university/thesis/datasets/MOTSynth_annotations/all",
        "split_path": None,
    }
    root_path = "/home/vy/university/thesis/datasets/MOTSynth"
    reader = MOTSynthReader(root_path, config)
    seq_id, frame_id = "045", 342

    # profile_absolute_egomotion_trajectory(reader, seq_id, 0, 50)
    # profile_relative_egomotion_trajectory(reader, seq_id, 0, 50)
    # profile_absolute_egomotion_transform(reader, seq_id, frame_id)
    profile_relative_egomotion_transform(reader, seq_id, frame_id)


if __name__ == "__main__":
    main()
