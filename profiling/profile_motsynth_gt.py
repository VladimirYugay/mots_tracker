""" profiling gt bb generation from motsynth """
import json

import numpy as np

from mots_tracker.readers import MOTSynthReader
from mots_tracker.vis_utils import M_COLORS, plot_image_boxes, plot_image_masks


def profile_empty_boxes(reader, seq_id, frame_id):
    """ Shows empty boxes with masks inconsistency """
    sample = reader.read_sample(seq_id, frame_id)
    image, boxes, box_ids = sample["image"], sample["boxes"], sample["box_ids"]
    masks, mask_ids = sample["masks"], sample["mask_ids"]
    non_empty_mask_ids = np.where(masks.sum(axis=(1, 2)) > 0)

    plot_image_masks(image, masks, mask_ids % len(M_COLORS))
    plot_image_boxes(image, box_ids % len(M_COLORS))
    plot_image_masks(image, masks[non_empty_mask_ids])
    plot_image_boxes(image, boxes[non_empty_mask_ids])
    # plot_image_boxes(image, boxes[np.where(masks.sum(axis=(1, 2)) == 0)])
    # plot_image_masks(image, masks[np.where(masks.sum(axis=(1, 2)) == 0)])


def main():
    """ visual profiling for generated motsynth bb """
    with open(
        "./configs/reader_configs/motsynth_reader_config.json", "r"
    ) as motsynth_config:
        config = json.load(motsynth_config)
    print(config)
    root_path = "/home/vy/university/thesis/datasets/MOTSynth"
    annotations_path = "/home/vy/university/thesis/datasets/MOTSynth"
    reader = MOTSynthReader(root_path, annotations_path, config)
    seq_id, frame_id = "001", 500
    profile_empty_boxes(reader, seq_id, frame_id)


if __name__ == "__main__":
    main()
