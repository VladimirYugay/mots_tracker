""" profiling MOTS dataset """
from mots_tracker import utils, vis_utils
from mots_tracker.readers import MOTSReader
from mots_tracker.trackers.tracker_helpers import depth_median_filter
from mots_tracker.utils import compute_mask_clouds, rgbd2ptcloud


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


def profile_ptclouds(reader, seq_id, frame_ids):
    """ Shows pedestrians point clouds """
    frames_pcds = []
    for frame_id in frame_ids:
        sample = reader.read_sample(seq_id, frame_id)
        clouds = compute_mask_clouds(sample, depth_median_filter, color_weight=0.5)
        frames_pcds.append(clouds)
    vis_utils.play_clouds_movement(frames_pcds)
    # vis_utils.plot_ptcloud(clouds)


def profile_scene(reader, seq_id, frame_id):
    """ Shows pedestrians point clouds """
    sample = reader.read_sample(seq_id, frame_id)
    scene_cloud = utils.rgbd2ptcloud(
        sample["image"], sample["depth"], sample["intrinsics"]
    )
    vis_utils.plot_ptcloud(scene_cloud)


def main():

    root_path = "/home/vy/university/thesis/datasets/MOTS"
    reader = MOTSReader(root_path)
    seq_id, frame_id = "MOTS20-11", 0

    # vis_utils.plot_image_masks(sample['image'], sample['masks'])
    # profile_masks(reader, seq_id, frame_id)
    profile_boxes(reader, seq_id, frame_id)


if __name__ == "__main__":
    main()
