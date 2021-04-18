""" module for reading motsynth  data """
#
# It's assumed to read data in the format on the server
# in /storage/user/brasoand/motsyn2 on 13.03.2021
#
# Annotations were converted to standard .txt file
# following format on https://motchallenge.net/data/MOT15/
#
# For box annotation conversion see
# lifting_mots/scripts/generate_motsynth_bb_annotations.py
#
# For mask annotation conversion see
# lifting_mots/scripts/generate_motsynth_mask_annotations.py
from configparser import ConfigParser
from pathlib import Path

import numpy as np

from mots_tracker import utils
from mots_tracker.readers import reader_helpers
from mots_tracker.readers.reader_helpers import read_mot_bb_file, read_mot_seg_file

DEFAULT_CONFIG = {
    "gt_path": "/home/vy/university/thesis/datasets/MOTSynth_annotations",
    "read_boxes": False,
    "read_masks": False,
    "resize_shape": None,
    "depth_path": None,
    "egomotion_path": None,
    "split_path": "split_0.6_0.2_0.2/train.txt",  # path_to_split_file
}

# taken from https://github.com/fabbrimatteo/JTA-Dataset
INTRINSICS = np.array([[1158, 0, 960], [0, 1158, 540], [0, 0, 1]], dtype=np.float64)


class MOTSynthReader(object):
    """ MOTSynth reader class """

    def __init__(self, root_path, config):
        """Reader constructor
        Args:
            root_path (str): path to frames folder with rgb images
            config (dict): config with reader setup options
        """
        self.config = DEFAULT_CONFIG.copy()
        self.config.update(config)
        self.gt_path = Path(self.config["gt_path"])
        self.root_path = Path(root_path)
        # keep cache only for one sequence since they're large
        self.cache = {}  # image names, box annotations, mask annotations
        self.sequence_info = self._init_sequence_info()

    def read_sample(self, seq_id, frame_id):
        """reads image and all annotations
        Args:
            seq_id (str): id of the sequence in 3-digit format
            frame_id (int): id of the frame
        Returns:{seq}
            dict (image, bb)
        """
        if seq_id not in self.cache:
            self._init_cache(seq_id)
        img_path = self.cache["img_names"][frame_id]
        boxes, box_ids, masks, mask_ids, raw_masks, image, depth, egomotion = [None] * 8
        if self.config["read_boxes"]:
            boxes, box_ids = self._read_bb(frame_id + 1)
        if self.config["read_masks"]:
            masks, mask_ids, raw_masks = self._read_seg_masks(seq_id, frame_id + 1)
        image = utils.load_image(img_path)
        if self.config["depth_path"] is not None:
            depth_path = self.gt_path / seq_id / self.config["depth_path"]
            depth_path = depth_path / "{:0>4d}".format(frame_id)
            if (
                self.config["depth_path"] == "depth"
            ):  # gt depth maps are images, not numpy
                depth = reader_helpers.load_motsynth_depth_image(
                    str(depth_path) + ".png"
                )
            else:
                depth = np.load(str(depth_path) + ".npy")
        if self.config["egomotion_path"] is not None:
            egomotion = self._read_egomotion(seq_id, frame_id)
        intrinsics = INTRINSICS
        if self.config["resize_shape"] is not None:
            width, height = image.size
            intrinsics = utils.scale_intrinsics(
                intrinsics, (height, width), self.config["resize_shape"]
            )
            if boxes is not None:
                boxes = utils.resize_boxes(
                    boxes, image.size, self.config["resize_shape"]
                )
            if image is not None:
                image = utils.resize_img(image, self.config["resize_shape"])
            if masks is not None:
                masks = utils.resize_masks(masks, self.config["resize_shape"])
        return {
            "boxes": boxes,
            "depth": depth,
            "box_ids": box_ids,
            "image": np.array(image),
            "masks": masks.astype(np.uint8) if masks is not None else masks,
            "raw_masks": raw_masks,
            "mask_ids": mask_ids,
            "intrinsics": intrinsics,
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
        masks_data, mask_strings = self.cache["masks"]
        masks_data = masks_data.copy()
        valid_mask = masks_data[:, 0] == frame_id
        relevant_ids = np.where(valid_mask)[0]
        masks_data = masks_data[valid_mask]
        height = self.sequence_info[seq_id]["img_height"]
        width = self.sequence_info[seq_id]["img_width"]
        raw_masks = [None] * relevant_ids.shape[0]
        masks = np.zeros((relevant_ids.shape[0], height, width), dtype=np.uint8)
        for i, rel_id in enumerate(relevant_ids):
            masks[i, ...] = utils.decode_mask(height, width, mask_strings[rel_id])
            raw_masks[i] = mask_strings[rel_id]
        # see notation here: https://www.vision.rwth-aachen.de/page/mots
        return masks, masks_data[:, 1].astype(np.uint64), raw_masks

    def _read_bb(self, frame_id):
        """read all bounding boxes for a given frame MOTS format
        Args:
            seq_id (str): sequence id
            frame_id (int): frame id
        Returns:
            boxes (ndarray), box_ids (ndarray): boxes with their ids
        """
        boxes = self.cache["boxes"].copy()
        frame_data = boxes[boxes[:, 0] == frame_id]
        box_ids = frame_data[:, 1].astype(np.uint64)
        frame_boxes = frame_data[:, [2, 3, 4, 5]]
        frame_boxes[:, 2] = frame_boxes[:, 0] + frame_boxes[:, 2]
        frame_boxes[:, 3] = frame_boxes[:, 1] + frame_boxes[:, 3]
        return frame_boxes, box_ids

    def _init_sequence_info(self):
        split_path = self.gt_path / ".." / self.config["split_path"]
        with open(str(split_path), "r") as file:
            seq_ids = set(file.read().splitlines())
        sequence_info = {}
        for info_file_path in (self.gt_path).glob("*"):
            parser = ConfigParser()
            parser.read(str(info_file_path / "seqinfo.ini"), encoding=None)
            seq_name = parser.get("Sequence", "name")
            if seq_name not in seq_ids:
                continue
            sequence_info[seq_name] = {
                "length": parser.getint("Sequence", "seqLength"),
                "img_width": parser.getint("Sequence", "imWidth"),
                "img_height": parser.getint("Sequence", "imHeight"),
            }
        return sequence_info

    def _init_cache(self, seq_id):
        """Initializes cache for a sequence
        Args:
             seq_id (str): sequence id in 3-digit format
        """
        imgs_path = self.root_path / "frames" / seq_id / "rgb"
        img_names = reader_helpers.read_file_names(imgs_path)
        img_names.sort()
        self.cache = {seq_id: seq_id, "img_names": img_names}
        if self.config["read_boxes"]:
            bb_path = self.gt_path / seq_id / "gt" / "gt.txt"
            self.cache["boxes"] = read_mot_bb_file(str(bb_path))

        if self.config["read_masks"]:
            mask_path = self.gt_path / seq_id / "gt" / "gt_masks.txt"
            self.cache["masks"] = read_mot_seg_file(mask_path)

    def _read_egomotion(self, seq_id, frame_id):
        """read rotation and translation of the camera from (frame_id - 1) to (frame_id)
        Args:
            seq_id (str): sequence id
            frame_id (int): frame id
        Returns:
            egomotion (ndarray): array representing rotation and translation
        """
        raise NotImplementedError
