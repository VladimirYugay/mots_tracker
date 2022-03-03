""" profiling MOT20 dataset """
import numpy as np

from mots_tracker import readers, utils, vis_utils
from mots_tracker.io_utils import get_instance, load_yaml

np.set_printoptions(suppress=True, precision=3)


def main():
    config_path = "./configs/debug_mot20_config.yaml"
    config = load_yaml(config_path)
    reader = get_instance(readers, "reader", config)
    seq_id, frame_id = "MOT20-04", 0
    sample = reader.read_sample(seq_id, frame_id)

    vis_utils.plot_image(sample["image"])
    vis_utils.plot_image(sample["depth"], image_type="depth")
    cloud = utils.rgbd2ptcloud(sample["image"], sample["depth"], sample["intrinsics"])
    vis_utils.plot_ptcloud(cloud)
    vis_utils.plot_image(sample["panoptic_mask"])
    vis_utils.plot_image(sample["panoptic_image"])


if __name__ == "__main__":
    main()
