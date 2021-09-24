""" module for reading mots ground truth data """
from configparser import ConfigParser
from pathlib import Path

import numpy as np

from mots_tracker import utils
from mots_tracker.readers import reader_helpers

DEFAULT_INTRINSICS = np.array(
    [[1539.579, 0, 940.66], [0, 1542.185, 556.894], [0, 0, 1]]
)
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

    def __init__(
        self,
        data_path,
        ann_path: str = "/home/vy/university/thesis/datasets/MOTS_annotations",
        depth_path: str = None,
        resize_shape: tuple = None,
        masks_path: str = "gt/gt.txt",
        boxes_path: str = "gt/gt_bb.txt",
        seq_ids: tuple = ("MOTS20-02", "MOTS20-05", "MOTS20-09", "MOTS20-11"),
        egomotion_path: str = None,
        mode: str = "train",
        conf_thres: float = 0.2,
    ):
        """constructor
        Args:
            data_path (str): directory containing both images and ground truth
            config (dict): configuration of the reader
        """
        self.data_path = Path(data_path)
        self.depth_path = depth_path
        self.ann_path = ann_path
        self.resize_shape = resize_shape
        self.masks_path = masks_path
        self.boxes_path = boxes_path
        self.seq_ids = seq_ids
        self.egomotion_path = egomotion_path
        self.mode = mode
        self.box_data_cache = {seq_id: None for seq_id in self.seq_ids}
        self.seg_data_cache = {seq_id: None for seq_id in self.seq_ids}
        self.egomotion_cache = {seq_id: None for seq_id in self.seq_ids}
        self.sequence_info = self._init_sequence_info()
        self.conf_thres = conf_thres

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
        img_path = reader_helpers.id2imgpath(
            seq_id, img_name, self.data_path / self.mode
        )
        # frame_id + 1 since frames are 1 indexed
        (
            boxes,
            box_ids,
            box_scores,
            masks,
            mask_ids,
            mask_scores,
            raw_masks,
            image,
            depth,
            egomotion,
        ) = [None] * 10
        if self.boxes_path:
            boxes, box_ids, box_scores = self._read_bb(seq_id, frame_id + 1)
        if self.masks_path:
            masks, mask_ids, raw_masks, mask_scores = self._read_seg_masks(
                seq_id, frame_id + 1
            )
        image = utils.load_image(img_path)
        if self.depth_path is not None:
            depth_name = "{:0>6d}".format(frame_id + 1) + ".npz"
            depth_path = (
                Path(self.ann_path) / self.mode / seq_id / self.depth_path / depth_name
            )
            depth = np.load(depth_path)["arr_0"]
            depth = utils.interpolate_depth(depth, image.size)
        if self.egomotion_path is not None:
            egomotion = self._read_egomotion(seq_id, frame_id)
        intrinsics = self.sequence_info[seq_id]["intrinsics"].copy()
        if self.resize_shape is not None:
            intrinsics = utils.scale_intrinsics(
                intrinsics, image.size, self.resize_shape
            )
            if boxes is not None:
                boxes = utils.resize_boxes(boxes, image.size, self.resize_shape)
            if image is not None:
                image = utils.resize_img(image, self.resize_shape)
            if masks is not None:
                masks = utils.resize_masks(masks, self.resize_shape)
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
            "mask_scores": mask_scores,
            "box_scores": box_scores,
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
            seg_path = self.data_path / self.mode / seq_id / self.masks_path
            self.seg_data_cache[seq_id] = reader_helpers.read_mot_seg_file(
                str(seg_path)
            )
        # data format: frame_id, obj_id, class_di h, w, mask string
        masks_data, mask_strings = self.seg_data_cache[seq_id]
        masks_data = masks_data.copy()
        mask_scores, mask_ids = np.zeros(masks_data.shape[0]), np.ones(
            masks_data.shape[0]
        )
        if "gt" in self.masks_path:
            mask_ids = masks_data[:, 1].astype(np.uint64) % 1000
        else:
            mask_scores = masks_data[:, 1]
        valid_mask = (
            (masks_data[:, 0] == frame_id)
            & (mask_ids > 0)
            & (mask_scores > self.conf_thres)
        )
        relevant_ids = np.where(valid_mask)[0]
        masks_data = masks_data[valid_mask]
        mask_ids = mask_ids[valid_mask]
        height, width = int(masks_data[0, 2]), int(masks_data[0, 3])
        raw_masks = [None] * relevant_ids.shape[0]
        masks = np.zeros((relevant_ids.shape[0], height, width), dtype=np.uint8)
        for i, rel_id in enumerate(relevant_ids):
            masks[i, ...] = utils.decode_mask(height, width, mask_strings[rel_id])
            raw_masks[i] = mask_strings[rel_id]
        # see notation here: https://www.vision.rwth-aachen.de/page/mots
        return masks, mask_ids, raw_masks, mask_scores

    def _read_egomotion(self, seq_id, frame_id):
        """read rotation and translation of the camera from (frame_id - 1) to (frame_id)
        Args:
            seq_id (str): sequence id
            frame_id (int): frame id
        Returns:
            egomotion (ndarray): array representing rotation and translation
        """
        static_seqs = ["MOTS20-01", "MOTS20-02", "MOTS20-07", "MOTS20-09"]

        if (
            "dummy" in str(self.egomotion_path)
            or seq_id in static_seqs
            or frame_id == 0
        ):
            return np.eye(4)
        if self.egomotion_cache[seq_id] is None:
            if "vo" not in str(self.egomotion_path):
                path = self.data_path / self.mode / seq_id / self.egomotion_path
                rot = np.load(str(path / "rotations.npy"), allow_pickle=True)
                trans = np.load(str(path / "translations.npy"), allow_pickle=True)
                transformations = np.zeros((rot.shape[0], 4, 4))
                transformations[:, :3, :] = np.concatenate((rot, trans), axis=2)
                transformations[:, -1, -1] = 1
                self.egomotion_cache[seq_id] = transformations
            elif "vo" in str(self.egomotion_path):
                path = self.data_path / self.mode / seq_id / self.egomotion_path
                self.egomotion_cache[seq_id] = reader_helpers.read_mots_vo_file(
                    str(path)
                )
            elif "mono+stereo_640x192" in str(self.egomotion_path):
                path = self.data_path / self.mode / seq_id / self.egomotion_path
                self.egomotion_cache[seq_id] = np.load(
                    str(path / "transformations.npy")
                )
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
            box_path = self.data_path / self.mode / seq_id / self.boxes_path
            self.box_data_cache[seq_id] = reader_helpers.read_mot_bb_file(str(box_path))
        boxes = self.box_data_cache[seq_id].copy()
        frame_data = boxes[boxes[:, 0] == frame_id]
        box_ids, box_scores = np.ones(frame_data.shape[0]), np.ones(frame_data.shape[0])
        if "gt" in self.boxes_path:
            box_ids = frame_data[:, 1].astype(np.uint16) % 1000
        else:
            box_scores = frame_data[:, 1]
        valid_mask = (box_ids > 0) & (box_scores > self.conf_thres)
        frame_boxes = frame_data[:, [2, 3, 4, 5]]
        frame_boxes[:, 2] = frame_boxes[:, 0] + frame_boxes[:, 2]
        frame_boxes[:, 3] = frame_boxes[:, 1] + frame_boxes[:, 3]
        return frame_boxes[valid_mask], box_ids[valid_mask], box_scores

    def _init_sequence_info(self):
        sequence_info = {idx: {} for idx in self.seq_ids}
        for idx in self.seq_ids:
            parser = ConfigParser()
            config_path = self.data_path / self.mode / idx / "seqinfo.ini"
            img_path = self.data_path / self.mode / idx / "img1"
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
