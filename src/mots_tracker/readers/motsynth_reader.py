""" module for reading motsynth  data """
# It's assumed to read data in the format on the server in /storage/user/brasoand/motsyn2 on 13.03.2021
# Annotations were converted to standard .txt file following format on https://motchallenge.net/data/MOT15/
# For box annotation conversion see lifting_mots/scripts/generate_motsynth_bb_annotations.py
# For mask annotation conversion see lifting_mots/scripts/generate_motsynth_mask_annotations.py
# from configparser import ConfigParser
from pathlib import Path

import numpy as np

from mots_tracker import utils
from mots_tracker.readers import reader_helpers

DEFAULT_CONFIG = {
    "read_boxes": True,
    "read_masks": True,
    "resize_shape": None,
    "depth_path": None,
    "egomotion_path": None,
}

# taken from https://github.com/fabbrimatteo/JTA-Dataset
INTRINSICS = np.array([[1158, 0, 960], [0, 1158, 540], [0, 0, 1]])


class MOTSynthReader(object):
    """ reader class """

    def __init__(self, root_path, gt_path, config):
        """constructor
        Args:
            root_path (str): directory containing both images and ground truth
            config (dict): configuration of the reader
        """
        self.config = DEFAULT_CONFIG.copy()
        self.config.update(config)
        self.gt_path = Path(
            gt_path
        )  # path to ground truth boxes, masks and depth (possibly)
        self.root_path = Path(root_path)
        # keep cache only for one sequence since they're large
        self.cache = {}  # image names, box annotations, mask annotations

    def read_sample(self, seq_id, frame_id):
        """reads image and all annotations
        Args:
            seq_id: id of the sequence
            frame_id: id of the frame
        Returns:{seq}
            dict (image, bb)
        """
        if seq_id not in self.cache:
            self.cache = {
                "seq_id": seq_id,
                "img_names": reader_helpers.read_file_names(
                    self.root_path / "frames" / "{:0>3d}".format(frame_id) / "rgb"
                ),
                "boxes": np.loadtxt(
                    str(self.gt_path / "box_annotations" / "{:0>3d}".format(frame_id)),
                    dtype=np.str,
                ),
                "masks": np.loadtxt(
                    str(self.gt_path / "mask_annotations" / "{:0>3d}".format(frame_id)),
                    dtype=np.str,
                ),
            }

        img_path = self.cache["img_names"][frame_id]
        img_name = reader_helpers.path2id(img_path)
        image = utils.load_image(img_path)
        boxes, box_ids, masks, mask_ids, raw_masks, image, depth, egomotion = [None] * 8
        if self.config["read_boxes"]:
            boxes, box_ids = self._read_bb(seq_id, frame_id + 1)
        if self.config["read_masks"]:
            masks, mask_ids, raw_masks = self._read_seg_masks(seq_id, frame_id + 1)
        if self.config["depth_path"] is not None:
            depth = np.load(
                reader_helpers.id2depthpath(
                    seq_id, img_name, self.root_path, self.config["depth_path"]
                )
            )
        if self.config["egomotion_path"] is not None:
            egomotion = self._read_egomotion(seq_id, frame_id)
        if self.config["resize_shape"] is not None:
            boxes = (
                utils.resize_boxes(boxes, image.size, self.config["resize_shape"])
                if boxes is not None
                else None
            )
            image = (
                utils.resize_img(image, self.config["resize_shape"])
                if image is not None
                else None
            )
            masks = (
                utils.resize_masks(masks, self.config["resize_shape"])
                if masks is not None
                else None
            )
        return {
            "boxes": boxes,
            "depth": depth,
            "box_ids": box_ids,
            "image": np.array(image),
            "masks": masks.astype(np.uint8),
            "raw_masks": raw_masks,
            "mask_ids": mask_ids,
            "intrinsics": INTRINSICS,
            "egomotion": egomotion,
        }

    def _read_seg_masks(self, seq_id, frame_id):
        """read all bounding boxes for a given frame
        Args:
            seq_id (str): sequence id
            frame_id (int): frame id
        Returns:
            masks (ndarray): binary object masks
        """
        # data format: frame_id, obj_id, class_id h, w, mask string
        seg_data = self.cache["masks"].copy()
        seg_data = seg_data[seg_data[:, 0] == str(frame_id)]
        mask_ids = seg_data[:, 1].astype(np.uint16)
        n_masks = mask_ids.shape[0]
        height, width = int(seg_data[:, 3][0]), int(seg_data[:, 4][0])
        masks, raw_masks = (
            np.zeros((n_masks, height, width), dtype=np.int),
            [None] * n_masks,
        )
        for i in range(n_masks):
            masks[
                i,
            ] = utils.decode_mask(seg_data[i][3], seg_data[i][4], seg_data[i][5])
            raw_masks[i] = seg_data[i][5]
        # see notation here: https://www.vision.rwth-aachen.de/page/mots
        return masks, mask_ids, np.array(raw_masks)

    def _read_bb(self, seq_id, frame_id):
        """read all bounding boxes for a given frame MOTS format
        Args:
            seq_id (str): sequence id
            frame_id (int): frame id
        Returns:
            boxes (ndarray), box_ids (ndarray): boxes with their ids
        """
        boxes = self.cache["boxes"].copy()
        frame_data = boxes[boxes[:, 0] == frame_id]
        box_ids = frame_data[:, 1].astype(np.uint16)
        frame_boxes = frame_data[:, [2, 3, 4, 5]]
        frame_boxes[:, 2] = frame_boxes[:, 0] + frame_boxes[:, 2]
        frame_boxes[:, 3] = frame_boxes[:, 1] + frame_boxes[:, 3]
        return frame_boxes, box_ids

    def _read_egomotion(self, seq_id, frame_id):
        """read rotation and translation of the camera from (frame_id - 1) to (frame_id)
        Args:
            seq_id (str): sequence id
            frame_id (int): frame id
        Returns:
            egomotion (ndarray): array representing rotation and translation
        """
        raise NotImplementedError
