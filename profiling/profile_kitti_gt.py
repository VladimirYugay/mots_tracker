from functools import partial
from pathlib import Path

from mots_tracker import utils, vis_utils
from mots_tracker.readers import KITTIReader
from mots_tracker.trackers.tracker_helpers import depth_median_filter
from mots_tracker.vis_utils import plot_image_boxes


def profile_boxes(sample):
    """ Visual test for boxes and their ids """
    plot_image_boxes(sample["image"], sample["boxes"], sample["box_ids"])


def profile_depth(sample):
    """ Visual test for depth """
    vis_utils.plot_image(sample["depth"], image_type="Depth")


def profile_resizing(sample):
    """ Visual test for depth """
    vis_utils.plot_image(sample["depth"], image_type="Depth")
    vis_utils.plot_image_boxes(sample["image"], sample["boxes"], sample["box_ids"])


def profile_clouds(reader, seq_id, frame_id):
    """ See the clouds """
    sample = reader.read_sample(seq_id, frame_id)
    cloud_filter = partial(depth_median_filter, radius=0.1)
    clouds = utils.compute_mask_clouds(sample, cloud_filter)
    vis_utils.plot_ptcloud(clouds, False)


def profile_scene(reader, seq_id, frame_id):
    """ Shows pedestrians point clouds """
    sample = reader.read_sample(seq_id, frame_id)
    scene_cloud = utils.rgbd2ptcloud(
        sample["image"], sample["depth"], sample["intrinsics"]
    )
    vis_utils.plot_ptcloud(scene_cloud)


if __name__ == "__main__":
    root_path = Path("/home/vy/university/thesis/datasets/KITTI")
    reader = KITTIReader(
        "/home/vy/university/thesis/datasets/KITTI/training",
        read_boxes=True,
        depth_path="adabins_for_mots_best",
    )
    seq_id, frame_id = "0007", 0
    sample = reader.read_sample(seq_id, frame_id)
    profile_depth(sample)
    profile_scene(reader, seq_id, frame_id)
    profile_boxes(sample)
