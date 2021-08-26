""" Module for reading KITTI dataset """
from pathlib import Path

import numpy as np

from mots_tracker import utils
from mots_tracker.readers import reader_helpers
from mots_tracker.readers.reader_helpers import read_file_names, read_kitti_bb_file


class KITTIReader(object):
    def __init__(
        self,
        data_path,
        read_boxes=True,
        depth_path=None,
        resize_shape=None,
        egomotion_path=None,
    ):
        self.read_boxes = read_boxes
        self.depth_path = depth_path
        self.resize_shape = resize_shape
        self.egomotion_path = egomotion_path
        self.data_path = Path(data_path)
        self.sequence_info = self._init_sequence_info()
        self.cache = {}

    def read_sample(self, seq_id: str, frame_id: int) -> dict:
        """Reads sample
        Args:
            seq_id: seqeunce id
            frame_id: frame id
        Returns:
            sample dictionary
        """
        if seq_id not in self.cache:
            self._init_cache(seq_id)

        intrinsics = self.cache["intrinsics"]
        intrinsics = intrinsics[0][:3, :3]  # temporal fix until clarification
        boxes, box_ids, image, depth, egomotion = [None] * 5
        img_path = self.cache["img_names"][frame_id]
        image = utils.load_image(img_path)
        if self.read_boxes:
            boxes, box_ids, obj_types = self._read_bb(frame_id)  # frames are 0 indexed
        if self.depth_path is not None:
            depth_path = self.data_path / self.depth_path / seq_id
            depth_path = str(depth_path) + "/{:06d}".format(frame_id) + ".npz"
            depth = np.load(depth_path)["arr_0"]
        if self.resize_shape is not None:
            width, height = image.size
            intrinsics = utils.scale_intrinsics(
                intrinsics, (height, width), self.resize_shape
            )
            if boxes is not None:
                boxes = utils.resize_boxes(boxes, image.size, self.resize_shape)
            if image is not None:
                image = utils.resize_img(image, self.resize_shape)

        return {
            "image": np.array(image),
            "boxes": boxes,
            "box_ids": box_ids,
            "intrinsics": intrinsics,
            "depth": depth,
            "egomotion": egomotion,
            "obj_types": obj_types,
        }

    def _read_bb(self, frame_id):
        """read all bounding boxes for a given frame MOTS format
        Args:
            frame_id (int): frame id
        Returns:
            boxes (ndarray), box_ids (ndarray), boj_types: boxes with their
            ids and types 1 - Car, 2 - Pedestrian
        """
        boxes, obj_types = self.cache["box_data"]
        frame_data = boxes[boxes[:, 0] == frame_id]
        box_ids = frame_data[:, 1].astype(np.uint64)
        frame_boxes = frame_data[:, [2, 3, 4, 5]]
        return frame_boxes, box_ids, obj_types

    def _init_cache(self, seq_id):
        """Initializes cache for a sequence
        Args:
             seq_id (str): sequence id in 4-digit format
        """
        img_path = self.data_path / "image_02" / seq_id
        img_names = reader_helpers.read_file_names(img_path)
        img_names.sort()
        bb_path = self.data_path / "label_02" / seq_id
        bb_path = str(bb_path) + ".txt"
        calib_path = self.data_path / "calib" / seq_id
        calib_path = str(calib_path) + ".txt"
        self.cache = {
            seq_id: seq_id,
            "box_data": read_kitti_bb_file(bb_path),
            "img_names": img_names,
            "intrinsics": reader_helpers.read_kitti_calib(calib_path),
        }

    def _init_sequence_info(self):
        """ Provides info about each of the sequences sequence """
        sequence_info = {}
        image_path = self.data_path / "image_02"
        for seq_id in image_path.iterdir():
            num_files = len(read_file_names(seq_id))
            seq_name = seq_id.parts[-1]
            sequence_info[seq_name] = {
                "length": num_files,
                "img_width": 1242,
                "img_height": 375,
            }
        return sequence_info
