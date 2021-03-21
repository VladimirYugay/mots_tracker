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
    "read_boxes": False,
    "read_masks": False,
    "resize_shape": None,
    "depth_path": None,
    "egomotion_path": None,
}

# taken from https://github.com/fabbrimatteo/JTA-Dataset
INTRINSICS = np.array([[1158, 0, 960], [0, 1158, 540], [0, 0, 1]])


class MOTSynthReader(object):
    """ MOTSynth reader class """

    def __init__(self, root_path, gt_path, config):
        """Reader constructor
        Args:
            root_path (str): path to frames folder with images and json annotations
            gt_path (str): path to motsynth bb, seg, depth annotations
            config (dict): config with reader setup options
        """
        self.config = DEFAULT_CONFIG.copy()
        self.config.update(config)
        self.gt_path = Path(gt_path)
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
            masks, mask_ids, raw_masks = self._read_seg_masks(frame_id + 1)
        image = utils.load_image(img_path)
        if self.config["depth_path"] is not None:
            depth = None  # not implemented
        if self.config["egomotion_path"] is not None:
            egomotion = self._read_egomotion(seq_id, frame_id)
        if self.config["resize_shape"] is not None:
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
            "intrinsics": INTRINSICS,
            "egomotion": egomotion,
        }

    def _read_seg_masks(self, frame_id):
        """read all bounding boxes for a given frame
        Args:
            seq_id (str): sequence id
            frame_id (int): frame id
        Returns:
            masks (ndarray): binary object masks
        """
        # data format: frame_id, obj_id, class_id h, w, mask string
        masks_data, mask_strings = self.cache["masks"]
        height, width = masks_data[0, 2], masks_data[0, 3]
        relevant_ids = np.where(masks_data[:, 0] == frame_id)[0]
        raw_masks = [None] * relevant_ids.shape[0]
        masks = np.zeros((relevant_ids.shape[0], height, width), dtype=np.uint8)
        for i, rel_id in enumerate(relevant_ids):
            masks[i, ...] = utils.decode_mask(height, width, mask_strings[rel_id])
            raw_masks[i] = mask_strings[rel_id]
        # see notation here: https://www.vision.rwth-aachen.de/page/mots
        return masks, relevant_ids, raw_masks

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
        sequence_info = {}
        for info_file_path in (self.gt_path / "sequences_info").glob("**/*"):
            parser = ConfigParser()
            parser.read(str(info_file_path))
            sequence_info[parser.get("Sequence", "name")] = parser.getint(
                "Sequence", "seqLength"
            )
        return sequence_info

    def _init_cache(self, seq_id):
        """Initializes cache for a sequence
        Args:
             seq_id (str): sequence id in 3-digit format
        """
        self.cache = {
            seq_id: seq_id,
            "img_names": sorted(
                reader_helpers.read_file_names(
                    self.root_path / "frames" / seq_id / "rgb"
                )
            ),
        }
        if self.config["read_boxes"]:
            bb_path = self.gt_path / "bb_annotations" / "{}.txt".format(seq_id)
            self.cache["boxes"] = read_mot_bb_file(str(bb_path))

        if self.config["read_masks"]:
            mask_path = self.gt_path / "mask_annotations" / "{}.txt".format(seq_id)
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
