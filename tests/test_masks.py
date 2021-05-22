""" Script for testing masks encoding and decoding """
import numpy as np

from mots_tracker import utils
from mots_tracker.readers import MOTSynthReader
from mots_tracker.trackers import tracker_helpers


def test_encoding(reader, seq_id, start, stop):
    for frame_id in range(start, stop):
        sample = reader.read_sample(seq_id, frame_id)
        height = reader.sequence_info[seq_id]["img_height"]
        width = reader.sequence_info[seq_id]["img_width"]
        encodings = [utils.encode_mask(mask) for mask in sample["masks"]]
        decodings = np.array(
            [
                utils.decode_mask(height, width, enc["counts"].decode(encoding="UTF-8"))
                for enc in encodings
            ]
        )
        assert np.all(sample["masks"] == decodings)


def test_encoding_iou(reader, seq_id, start, stop):
    for frame_id in range(start, stop - 1):
        sample_left = reader.read_sample(seq_id, frame_id)
        sample_right = reader.read_sample(seq_id, frame_id + 1)
        str_iou = tracker_helpers.iou_string_masks(
            sample_right["masks"], sample_left["masks"]
        )
        iou = tracker_helpers.iou_masks(sample_right["masks"], sample_left["masks"])
        assert np.all(str_iou == iou)


if __name__ == "__main__":
    config = {
        "depth_path": "gt_depth",
        "egomotion_path": "egomotion",
        "read_masks": True,
        "read_boxes": False,
        "gt_path": "/home/vy/university/thesis/datasets/MOTSynth_annotations/all",
        "split_path": None,
    }
    root_path = "/home/vy/university/thesis/datasets/MOTSynth"
    reader = MOTSynthReader(root_path, config)
    seq_id = "045"
    # test_encoding(reader, seq_id, 0, 5)
    test_encoding_iou(reader, seq_id, 0, 2)
