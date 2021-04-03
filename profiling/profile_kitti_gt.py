from pathlib import Path

from mots_tracker.readers import KITTIReader
from mots_tracker.vis_utils import plot_image_boxes


def profile_boxes(sample):
    """ Visual test for boxes and their ids """
    plot_image_boxes(sample["image"], sample["boxes"], sample["box_ids"])


if __name__ == "__main__":
    root_path = Path("/home/vy/university/thesis/datasets/KITTI")
    reader = KITTIReader(root_path / "training", {})
    seq_id, frame_id = 0, 100
    sample = reader.read_sample(seq_id, frame_id)
    # profile_boxes(sample)
    # print(reader.sequence_info)
