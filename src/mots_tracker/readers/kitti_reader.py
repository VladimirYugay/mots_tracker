""" Module for reading KITTI dataset """
from pathlib import Path

import numpy as np

from mots_tracker import utils
from mots_tracker.readers import reader_helpers
from mots_tracker.readers.reader_helpers import read_file_names, read_kitti_bb_file

DEFAULT_CONFIG = {
    "read_boxes": True,
    "resize_shape": None,
    "depth_path": None,
    "egomotion_path": None,
}


class KITTIReader(object):
    def __init__(self, root_path, config):
        self.config = DEFAULT_CONFIG.copy()
        self.config.update(config)
        self.root_path = Path(root_path)
        self.sequence_info = self._init_sequence_info()
        self.cache = {}

    def read_sample(self, seq_id, frame_id):
        """Reads sample
        Args:
            seq_id (int): seqeunce id
            frame_id (int): frame id
        Returns:
            dict: sample
        """
        if seq_id not in self.cache:
            self._init_cache(seq_id)

        img_path = self.cache["img_names"][frame_id]
        image = utils.load_image(img_path)
        if self.config["read_boxes"]:
            boxes, box_ids = self._read_bb(frame_id)  # frames are 0 indexed

        return {
            "image": image,
            "boxes": boxes,
            "box_ids": box_ids,
            "intrinsics": self.cache["intrinsics"],
        }

    def _read_bb(self, frame_id):
        """read all bounding boxes for a given frame MOTS format
        Args:
            seq_id (str): sequence id
            frame_id (int): frame id
        Returns:
            boxes (ndarray), box_ids (ndarray): boxes with their ids
        """
        boxes = self.cache["box_data"].copy()
        frame_data = boxes[boxes[:, 0] == frame_id]
        box_ids = frame_data[:, 1].astype(np.uint64)
        frame_boxes = frame_data[:, [2, 3, 4, 5]]
        return frame_boxes, box_ids

    def _init_cache(self, seq_id):
        """Initializes cache for a sequence
        Args:
             seq_id (str): sequence id in 3-digit format
        """
        img_path = self.root_path / "image_02" / "{:04d}".format(seq_id)
        img_names = reader_helpers.read_file_names(img_path)
        img_names.sort()
        bb_path = self.root_path / "label_02" / "{:04d}".format(seq_id)
        bb_path = str(bb_path) + ".txt"
        calib_path = self.root_path / "calib" / "{:04d}".format(seq_id)
        calib_path = str(calib_path) + ".txt"
        self.cache = {
            seq_id: seq_id,
            "box_data": read_kitti_bb_file(bb_path),
            "img_names": img_names,
            "intrinsics": reader_helpers.read_kitti_calib(calib_path),
        }

    def _init_sequence_info(self):
        sequence_info = {}
        image_path = self.root_path / "image_02"
        for seq_id in image_path.iterdir():
            num_files = len(read_file_names(seq_id))
            seq_name = seq_id.parts[-1]
            sequence_info[seq_name] = {"length": num_files}
        return sequence_info
