""" profiling MOTS dataset """
from functools import partial

import numpy as np

from mots_tracker import readers, utils, vis_utils
from mots_tracker.io_utils import get_instance, load_yaml
from mots_tracker.trackers.tracker_helpers import depth_median_filter
from mots_tracker.utils import rgbd2ptcloud

def profile_image(sample):
    vis_utils.plot_image(sample["image"])

def main():
    config_path = "./configs/debug_config.yaml"
    config = load_yaml(config_path)
    reader = get_instance(readers, "reader", config)
    seq_id, frame_id = "MOT16-03", 295
    sample = reader.read_sample(seq_id, frame_id)
    print(sample.keys())    
    profile_image(sample)


if __name__ == "__main__":
    main()
