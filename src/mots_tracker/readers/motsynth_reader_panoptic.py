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
from argparse import Namespace
import pickle
import itertools
import json

from mots_tracker import utils
from mots_tracker.readers import reader_helpers
from mots_tracker.readers.reader_helpers import (
    read_mot_bb_file,
    read_mot_seg_file,
    read_motsynth_2d_keypoints_file,
    read_motsynth_3d_keypoints_file,
    read_motsynth_egomotion_file,
)

# taken from https://github.com/fabbrimatteo/JTA-Dataset
INTRINSICS = np.array([[1158, 0, 960], [0, 1158, 540], [0, 0, 1]], dtype=np.float64)


class MOTSynthReaderPanoptic(object):
    """ MOTSynth reader class """

    def __init__(
        self,
        data_path: str,
        ann_path: str = "/home/vy/university/thesis/datasets/MOTSynth_annotations",
        boxes_path: str = "gt/gt.txt",
        masks_path: str = "gt/gt_masks.txt",
        resize_shape: tuple = None,
        depth_path: str = "gt_depth_new",
        egomotion_path: str = "egomotion",
        keypoints_2d_path: str = None,
        keypoints_3d_path: str = None,
        num_kpts=22,
        split_path: str = None,

        panoptic_path: str = None,
        catgoery2class_json_path=None,
        metadata_path=None,
        include_cats=None,
        exclude_cats=None,
    ):
        """Reader constructor
        Args:
            data_path (str): path to frames folder with rgb images
            config (dict): config with reader setup options
        """
        self.data_path = Path(data_path)
        self.ann_path = Path(ann_path)
        self.boxes_path = boxes_path
        self.masks_path = masks_path
        self.resize_shape = resize_shape
        self.depth_path = depth_path
        self.egomotion_path = egomotion_path
        self.split_path = split_path
        self.num_kpts = num_kpts
        self.keypoints_2d_path = keypoints_2d_path
        self.keypoints_3d_path = keypoints_3d_path
        # keep cache only for one sequence since they're large
        self.cache = {}  # image names, box annotations, mask annotations
        self.sequence_info = self._init_sequence_info()

        self.panoptic_path = panoptic_path
        self.catgoery2class_json_path = catgoery2class_json_path
        self.metadata_path = metadata_path
        self.include_cats = include_cats
        self.exclude_cats = exclude_cats

        self.color_ids = []
        self.category2class = {}
        self.meta_data = None
        self.color2coco = {}
        self.category_colors = {}

        self._init_metadata()
        self._init_color_ids()
        self._init_category2class()
        self._init_color2coco()
        self._init_category_colors()      

    def _init_color_ids(self, step=1000000):
        """ Initializes color ids """
        palette = list(itertools.product(np.arange(1, 256), repeat=3))
        self.color_ids = [palette[i::step] for i in range(step)]

    def _init_category2class(self):
        """ Initializes class to category mapping """
        with open(self.catgoery2class_json_path) as json_file:
            self.classes2category = json.load(json_file)

    def _init_metadata(self):
        """ Intializes meta_data on the panoptic classes """
        with open(self.metadata_path, "rb") as pkl_file:
            self.meta_data = Namespace(**pickle.load(pkl_file))

    def _init_color2coco(self):
        """ Initializes mapping of color to coco """
        self.color_to_coco = {}
        for stuff_class, stuff_color in zip(
            self.meta_data.stuff_classes, self.meta_data.stuff_colors
        ):
            color = " ".join(str(e) for e in stuff_color)
            self.color2coco[color] = stuff_class
        for thing_class, thing_color in zip(
            self.meta_data.thing_classes, self.meta_data.thing_colors
        ):
            color = " ".join(str(e) for e in thing_color)
            self.color2coco[color] = thing_class

    def _init_category_colors(self):
        """ Initializes category colors dict """
        self.category_colors = {
            "sky": np.array([70, 130, 180]),
            "pedestrian": np.array([255, 0, 0]),
            "other": np.array([255, 215, 0]),
            "occluder_moving": np.array([0, 100, 100]),
            "occluder_static": np.array([0, 0, 230]),
            "building": np.array([0, 255, 0]),
            "road": np.array([50, 50, 50]),
            "pavement": np.array([100, 100, 100]),
            "ground": np.array([10, 200, 10]),
        }


    def read_panoptic_img(self, seq_id, frame_id):
        """Reads depth map file
        Args:
            seq_id: sequence ud
            frame_id: frame id
        Returns:
            np.ndarray: float depth map
        """
        if self.panoptic_path is None:
            return None, None

        img_path = self.sequence_info[seq_id]["img_paths"][frame_id]
        panoptic_path = str(
            Path(self.panoptic_path)
            / seq_id
            / str(img_path.parts[-1]).replace(".jpg", ".png")
        )
        img = utils.load_image(panoptic_path)
        img = np.array(img)
        colors = np.unique(img.reshape(-1, img.shape[2]), axis=0)
        panoptic_img = np.zeros_like(img)
        panoptic_mask = np.zeros((img.shape[0], img.shape[1]))
        for color in colors:
            color_str = " ".join(str(e) for e in color)
            coco_class = self.color2coco.get(color_str, None)
            if coco_class is None:
                continue
            category = self.classes2category[coco_class]
            if self.include_cats and category not in self.include_cats:
                continue
            if self.exclude_cats and category in self.exclude_cats:
                continue
            final_color = self.category_colors[category]
            color_indices = np.where(np.all(img == color, axis=-1))
            panoptic_img[color_indices] = final_color
            panoptic_mask[color_indices] = self.categories.index(category) + 1
        return panoptic_img, panoptic_mask 


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
        boxes, box_ids, boxes_confidence = self._read_bb(frame_id + 1)
        masks, mask_ids, raw_masks, masks_confidence = self._read_seg_masks(
            seq_id, frame_id + 1
        )
        image = utils.load_image(img_path)
        depth = self._read_depth(seq_id, frame_id, self.resize_shape)
        egomotion = self.cache["egomotion"][frame_id]
        intrinsics = INTRINSICS
        _, keypoints_2d = self._read_kpts(frame_id + 1, is_3d=False)
        _, keypoints_3d = self._read_kpts(frame_id + 1, is_3d=True)

        panoptic_image, panoptic_mask = self.read_panoptic_img(seq_id, frame_id)

        if self.resize_shape is not None:
            width, height = image.size
            intrinsics = utils.scale_intrinsics(
                intrinsics, (width, height), self.resize_shape
            )
            if boxes is not None:
                boxes = utils.resize_boxes(boxes, image.size, self.resize_shape)
            if image is not None:
                image = utils.resize_img(image, self.resize_shape)
            if masks is not None:
                masks = utils.resize_masks(masks, self.resize_shape)

        return {
            "boxes": boxes,
            "box_ids": box_ids,
            "boxes_confidence": boxes_confidence,
            "depth": depth,
            "image": np.array(image),
            "masks": masks.astype(np.uint8) if masks is not None else masks,
            "mask_ids": mask_ids,
            "keypoints_2d": keypoints_2d,
            "keypoints_3d": keypoints_3d,
            "masks_confidence": masks_confidence,
            "raw_masks": raw_masks,
            "intrinsics": intrinsics,
            "egomotion": egomotion,
            "panoptic_image": panoptic_image,
            "panoptic_mask": panoptic_mask
        }

    def _read_seg_masks(self, seq_id: str, frame_id: int) -> tuple:
        """read all bounding boxes for a given frame
        Args:
            seq_id: sequence id
            frame_id: frame id
        Returns:
            masks, masks_ids, raw_masks, masks_confidence: numpy arrays
        """
        # data format: frame_id, obj_id, class_id h, w,
        # mask string (or confidence score instead of obj_id))
        if self.masks_path is None:
            return None, None, None, None
        masks_data, mask_strings = self.cache["masks"]
        masks_data = masks_data.copy()
        mask_ids = masks_data[:, 1].astype(np.int64)
        confidence = np.ones(masks_data.shape[0])
        if "gt" not in self.masks_path:
            confidence = masks_data[:, 1].copy()
        valid_mask = (masks_data[:, 0] == frame_id) & (confidence > 0.3)
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
        return masks, mask_ids, raw_masks, confidence

    def _read_bb(self, frame_id: int) -> tuple:
        """read all bounding boxes for a given frame MOTS format
        Args:
            frame_id (int): frame id
        Returns:
            boxes, box_ids, confidence: boxes with their ids and confidence
        """
        if self.boxes_path is None:
            return None, None, None
        boxes = self.cache["boxes"].copy()
        confidence = np.ones(boxes.shape[0])
        # dependent on gt or other model inference, `obj_id` plays a role of
        # an object id and of a confidence score correspondingly
        if "gt" not in self.boxes_path:
            confidence = boxes[:, 1].copy()
        frame_data = boxes[(boxes[:, 0] == frame_id) & (confidence > 0.3)]
        frame_boxes = frame_data[:, [2, 3, 4, 5]]
        frame_boxes[:, 2] = frame_boxes[:, 0] + frame_boxes[:, 2]
        frame_boxes[:, 3] = frame_boxes[:, 1] + frame_boxes[:, 3]
        return frame_boxes, frame_data[:, 1].astype(np.int64), boxes[:, 1]

    def _read_kpts(self, frame_id: int, is_3d=False) -> tuple:
        """read all kpts (2D or 3D) for a given frame
        Args:
            frame_id (int): frame id
        Returns:
            keypoints, keypoint_ids: keypoints with their ids
        """
        if self.keypoints_2d_path is None:
            return None, None
        keypoints = self.cache["keypoints_2d"].copy()
        if is_3d:
            keypoints = self.cache["keypoints_3d"].copy()
        frame_kpts = keypoints[keypoints[:, 0] == frame_id]
        kpt_ids = frame_kpts[:, 1]
        kpts = frame_kpts[:, 2:].reshape(kpt_ids.shape[0], self.num_kpts, -1)
        return kpt_ids, kpts

    def _init_sequence_info(self):
        """ Initializes information about the sequences """
        seq_ids = set(pth.parts[-1] for pth in self.ann_path.glob("*"))
        if self.split_path is not None:
            split_path = self.ann_path / ".." / self.split_path
            with open(str(split_path), "r") as file:
                seq_ids = set([line.strip() for line in file.readlines()])
        sequence_info = {}
        for info_file_path in (self.ann_path).glob("*"):
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
        imgs_path = self.data_path / "frames" / seq_id
        img_names = reader_helpers.read_file_names(imgs_path)
        img_names.sort()
        self.cache = {seq_id: seq_id, "img_names": img_names}

        if self.boxes_path is not None:
            bb_path = self.ann_path / seq_id / self.boxes_path
            self.cache["boxes"] = read_mot_bb_file(str(bb_path))

        if self.masks_path is not None:
            mask_path = self.ann_path / seq_id / self.masks_path
            self.cache["masks"] = read_mot_seg_file(mask_path)

        if self.keypoints_2d_path is not None:
            keypoints_path = self.ann_path / seq_id / self.keypoints_2d_path
            self.cache["keypoints_2d"] = read_motsynth_2d_keypoints_file(keypoints_path)

        if self.keypoints_3d_path is not None:
            keypoints_path = self.ann_path / seq_id / self.keypoints_3d_path
            self.cache["keypoints_3d"] = read_motsynth_3d_keypoints_file(keypoints_path)

        if self.egomotion_path == "egomotion":  # gt path is egomotion
            egomotion_path = self.ann_path / seq_id / "gt" / "egomotion.txt"
            egomotion = read_motsynth_egomotion_file(egomotion_path)
            self.cache["egomotion"] = egomotion
        else:
            self.cache["egomotion"] = np.repeat(
                np.eye(4)[
                    None,
                ],
                1800,
                axis=0,
            )

    def _read_depth(self, seq_id: str, frame_id: int, size: tuple = None):
        """read rotation and translation of the camera from origin to the current frame
        Args:
            seq_id: sequence id
            frame_id: frame id
        Returns:
            depth: depth map
        """
        if self.depth_path is None:
            return None
        depth_path = self.ann_path / seq_id / self.depth_path
        depth_path = depth_path / "{:0>4d}".format(frame_id)
        if "gt" in self.depth_path:
            depth = reader_helpers.load_motsynth_depth_image(
                str(depth_path) + ".png", self.resize_shape
            )
        else:
            depth = np.load(str(depth_path) + ".npz")["arr_0"]
            depth = np.clip(depth, 0, 100)
            if size is not None:
                depth = utils.interpolate_depth(depth, tuple(size))
        return depth
