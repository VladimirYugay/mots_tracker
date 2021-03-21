""" profiling MOTS dataset """
from mots_tracker import vis_utils
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


def profile_ptclouds(reader, seq_id, frame_id):
    """ Shows pedestrians point clouds """
    sample = reader.read_sample(seq_id, frame_id)
    clouds = compute_mask_clouds(sample, depth_median_filter, color_weight=0.5)
    vis_utils.plot_ptcloud(clouds)


def main():
    config = {"depth_path": "depth_diw_416_128", "resize_shape": [416, 128]}
    root_path = "/home/vy/university/thesis/datasets/MOTS/train"
    reader = MOTSReader(root_path, config)
    seq_id, frame_id = "MOTS20-02", 0
    profile_intrinsics(reader, seq_id, frame_id)


if __name__ == "__main__":
    main()
