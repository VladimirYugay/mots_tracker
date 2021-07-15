""" profiling MOTS dataset """
from functools import partial

from mots_tracker import readers, utils, vis_utils
from mots_tracker.io_utils import get_instance, load_yaml
from mots_tracker.trackers.tracker_helpers import depth_median_filter
from mots_tracker.utils import rgbd2ptcloud


def profile_intrinsics(reader, seq_id, frame_id):
    """ Shows empty boxes with masks inconsistency """
    sample = reader.read_sample(seq_id, frame_id)
    image = sample["image"]
    depth, intrinsics = sample["depth"], sample["intrinsics"]
    cloud = rgbd2ptcloud(image, depth, intrinsics)
    vis_utils.plot_ptcloud(cloud)


def profile_masks(reader, seq_id, frame_id):
    """ Shows image with masks and their ids """
    sample = reader.read_sample(seq_id, frame_id)
    image = sample["image"]
    masks, mask_ids = sample["masks"], sample["mask_ids"]
    vis_utils.plot_image_masks(image, masks, mask_ids)


def profile_boxes(reader, seq_id, frame_id):
    """ Shows image with boxes and their ids """
    sample = reader.read_sample(seq_id, frame_id)
    image, boxes, box_ids = sample["image"], sample["boxes"], sample["box_ids"]
    vis_utils.plot_image_boxes(image, boxes, box_ids)


def profile_depth(reader, seq_id, frame_id):
    """ Shows image with boxes and their ids """
    sample = reader.read_sample(seq_id, frame_id)
    vis_utils.plot_image(sample["depth"], image_type="Depth")


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


def main():
    config_path = "./configs/2dbb_tracker_config.yaml"
    config = load_yaml(config_path)
    reader = get_instance(readers, "reader", config)
    seq_id, frame_id = "MOTS20-02", 0

    # vis_utils.plot_image_masks(sample['image'], sample['masks'])
    # profile_masks(reader, seq_id, frame_id)
    # profile_boxes(reader, seq_id, frame_id)
    # profile_depth(reader, seq_id, frame_id)
    # profile_scene(reader, seq_id, frame_id)
    profile_clouds(reader, seq_id, frame_id)


if __name__ == "__main__":
    main()
