""" module for reading mots ground truth data """
from configparser import ConfigParser
from pathlib import Path

import numpy as np

from mots_tracker import utils
from mots_tracker.readers import reader_helpers

SEQUENCE_IDS = ("MOTS20-02", "MOTS20-05", "MOTS20-09", "MOTS20-11")

DEFAULT_CONFIG = {
    "depth_path": None,
    "read_boxes": True,
    "read_masks": True,
    "resize_shape": None,
    "vis_threshold": 0.25,
    "seq_ids": ("MOTS20-02", "MOTS20-05", "MOTS20-09", "MOTS20-11"),
    "bbs_file_name": "gt_bb.txt",
    "masks_file_name": "gt.txt",
    "egomotion_path": "egomotion_ootb",
}
DEFAULT_INTRINSICS = np.array(
    [[1224.369, 0, 925.2], [0, 674.325, 581.681], [0, 0, 1]]
)  # thumbnail
INTRINSICS = {
    "MOTS20-02": DEFAULT_INTRINSICS,  # gt unknown
    "MOTS20-05": np.array([[501.167, 0, 307.514], [0, 501.049, 229.744], [0, 0, 1]]),
    "MOTS20-09": np.array([[1539.579, 0, 940.66], [0, 1542.185, 556.894], [0, 0, 1]]),
    "MOTS20-11": np.array([[1539.579, 0, 940.66], [0, 1542.185, 556.894], [0, 0, 1]]),
    "MOTS20-01": DEFAULT_INTRINSICS,  # gt unknown
    "MOTS20-06": np.array([[499.759, 0, 307.488], [0, 500.335, 230.462], [0, 0, 1]]),
    "MOTS20-07": np.array([[1539.579, 0, 940.66], [0, 1542.185, 556.894], [0, 0, 1]]),
    "MOTS20-12": np.array([[1539.579, 0, 940.66], [0, 1542.185, 556.894], [0, 0, 1]]),
}


class MOTSReader(object):
    """ reader class """

    def __init__(self, root_path, config):
        """constructor
        Args:
            root_path (str): directory containing both images and ground truth
            config (dict): configuration of the reader
        """
        self.config = DEFAULT_CONFIG.copy()
        self.config.update(config)
        self.root_path = Path(root_path)
        self.box_data_cache = {seq_id: None for seq_id in self.config["seq_ids"]}
        self.seg_data_cache = {seq_id: None for seq_id in self.config["seq_ids"]}
        self.egomotion_cache = {seq_id: None for seq_id in self.config["seq_ids"]}
        self.sequence_info = self._init_sequence_info(self.config["seq_ids"])

    def read_sample(self, seq_id, frame_id):
        """reads image and all annotations
        Args:
            seq_id: id of the sequence
            frame_id: id of the frame
        Returns:{seq}
            dict (image, bb)
        """
        assert seq_id in self.sequence_info.keys()
        assert 0 <= frame_id < len(self.sequence_info[seq_id]["img_names"])
        img_name = self.sequence_info[seq_id]["img_names"][frame_id]
        img_path = reader_helpers.id2imgpath(seq_id, img_name, self.root_path)
        # frame_id + 1 since frames are 1 indexed
        boxes, box_ids, masks, mask_ids, raw_masks, image, depth, egomotion = [None] * 8
        if self.config["read_boxes"]:
            boxes, box_ids = self._read_bb(seq_id, frame_id + 1)
        if self.config["read_masks"]:
            masks, mask_ids, raw_masks = self._read_seg_masks(seq_id, frame_id + 1)
        image = utils.load_image(img_path)
        if self.config["depth_path"] is not None:
            depth = np.load(
                reader_helpers.id2depthpath(
                    seq_id, img_name, self.root_path, self.config["depth_path"]
                )
            )
        if self.config["egomotion_path"] is not None:
            egomotion = self._read_egomotion(seq_id, frame_id)
        intrinsics = self.sequence_info[seq_id]["intrinsics"].copy()
        if self.config["resize_shape"] is not None:
            print((image.size[1], image.size[0]), self.config["resize_shape"])
            intrinsics = utils.scale_intrinsics(
                intrinsics, image.size, self.config["resize_shape"]
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
        # cache ground truth txt file
        if self.seg_data_cache[seq_id] is None:
            seg_path = self.root_path / seq_id / "gt" / self.config["masks_file_name"]
            self.seg_data_cache[seq_id] = np.loadtxt(str(seg_path), dtype=np.str)
        # data format: frame_id, obj_id, class_di h, w, mask string
        seg_data = self.seg_data_cache[seq_id].copy()
        seg_data = seg_data[seg_data[:, 0] == str(frame_id)]
        mask_ids = seg_data[:, 1].astype(np.uint16) % 1000
        seg_data = seg_data[mask_ids > 0]  # ignore invalid masks
        mask_ids = mask_ids[mask_ids > 0]
        n_masks = mask_ids.shape[0]
        height, width = (
            self.sequence_info[seq_id]["img_height"],
            self.sequence_info[seq_id]["img_width"],
        )
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

    def _read_egomotion(self, seq_id, frame_id):
        """read rotation and translation of the camera from (frame_id - 1) to (frame_id)
        Args:
            seq_id (str): sequence id
            frame_id (int): frame id
        Returns:
            egomotion (ndarray): array representing rotation and translation
        """
        if self.egomotion_cache[seq_id] is None:
            rotations_path = (
                self.root_path
                / seq_id
                / self.config["egomotion_path"]
                / "rotations.npy"
            )
            translations_path = (
                self.root_path
                / seq_id
                / self.config["egomotion_path"]
                / "translations.npy"
            )
            rot, trans = (
                np.load(str(rotations_path))[
                    :,
                    0,
                ],
                np.load(str(translations_path))[
                    :,
                    0,
                ],
            )
            transformations = np.zeros((rot.shape[0], 4, 4))
            transformations[:, :3, :] = np.concatenate((rot, trans[..., None]), axis=2)
            transformations[:, -1, -1] = 1
            self.egomotion_cache[seq_id] = transformations
        if frame_id == 0:
            return np.eye(4)
        return self.egomotion_cache[seq_id][frame_id - 1]

    def _read_bb(self, seq_id, frame_id):
        """read all bounding boxes for a given frame MOTS format
        Args:
            seq_id (str): sequence id
            frame_id (int): frame id
        Returns:
            boxes (ndarray), box_ids (ndarray): boxes with their ids
        """
        if self.box_data_cache[seq_id] is None:
            box_path = self.root_path / seq_id / "gt" / self.config["bbs_file_name"]
            self.box_data_cache[seq_id] = np.loadtxt(str(box_path), delimiter=",")
        boxes = self.box_data_cache[seq_id].copy()
        frame_data = boxes[boxes[:, 0] == frame_id]
        box_ids = frame_data[:, 1].astype(np.uint16) % 1000
        frame_boxes = frame_data[:, [2, 3, 4, 5]]
        frame_boxes[:, 2] = frame_boxes[:, 0] + frame_boxes[:, 2]
        frame_boxes[:, 3] = frame_boxes[:, 1] + frame_boxes[:, 3]
        return frame_boxes, box_ids

    def _init_sequence_info(self, sequence_ids):
        sequence_info = {idx: {} for idx in sequence_ids}
        for idx in sequence_ids:
            parser = ConfigParser()
            config_path = self.root_path / idx / "seqinfo.ini"
            img_path = self.root_path / idx / "img1"
            parser.read(config_path)
            sequence_info[idx] = {
                "length": int(parser["Sequence"]["seqLength"]),
                "img_width": int(parser["Sequence"]["imWidth"]),
                "img_height": int(parser["Sequence"]["imHeight"]),
                "img_names": sorted(
                    [reader_helpers.path2id(path) for path in img_path.glob("*.jpg")]
                ),
                "intrinsics": INTRINSICS[idx],
            }
        return sequence_info
