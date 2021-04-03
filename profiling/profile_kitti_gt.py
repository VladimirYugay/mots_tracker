from pathlib import Path

from mots_tracker import vis_utils
from mots_tracker.readers import KITTIReader
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


if __name__ == "__main__":
    root_path = Path("/home/vy/university/thesis/datasets/KITTI")
    config = {"depth_path": "kitti"}
    reader = KITTIReader(root_path / "training", config)
    seq_id, frame_id = 15, 100
    sample = reader.read_sample(seq_id, frame_id)
    profile_resizing(sample)
    # profile_depth(sample)
    # profile_boxes(sample)
    # print(reader.sequence_info)
