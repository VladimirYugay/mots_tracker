""" module for reading mots ground truth data """
import itertools
import json
import pickle
from argparse import Namespace
from configparser import ConfigParser
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from mots_tracker import utils

MIN_DEPTH = 0.1  # in meters
MAX_DEPTH = 100  # in meters

DEFAULT_INTRINSICS = np.array(
    [[1539.579, 0, 940.66], [0, 1542.185, 556.894], [0, 0, 1]]
)

INTRINSICS = {
    "MOT16-01": DEFAULT_INTRINSICS,
    "MOT16-02": DEFAULT_INTRINSICS,
    "MOT16-03": DEFAULT_INTRINSICS,
    "MOT16-04": DEFAULT_INTRINSICS,
    "MOT16-05": np.array([[501.167, 0, 307.514], [0, 501.049, 229.744], [0, 0, 1]]),
    "MOT16-06": np.array([[499.759, 0, 307.488], [0, 500.335, 230.462], [0, 0, 1]]),
    "MOT16-07": np.array([[1539.579, 0, 940.66], [0, 1542.185, 556.894], [0, 0, 1]]),
    "MOT16-08": DEFAULT_INTRINSICS,
    "MOT16-09": DEFAULT_INTRINSICS,
    "MOT16-10": DEFAULT_INTRINSICS,
    "MOT16-11": np.array([[1539.579, 0, 940.66], [0, 1542.185, 556.894], [0, 0, 1]]),
    "MOT16-12": np.array([[1539.579, 0, 940.66], [0, 1542.185, 556.894], [0, 0, 1]]),
    "MOT16-13": DEFAULT_INTRINSICS,
    "MOT16-14": DEFAULT_INTRINSICS,
}

DYNAMIC_SEQUENCES = {
    "MOT16-01": False,
    "MOT16-02": False,
    "MOT16-03": False,
    "MOT16-04": False,
    "MOT16-05": True,
    "MOT16-06": True,
    "MOT16-07": True,
    "MOT16-08": False,
    "MOT16-09": False,
    "MOT16-10": True,
    "MOT16-11": True,
    "MOT16-12": True,
    "MOT16-13": True,
    "MOT16-14": True,
}

TAG_FLOAT = 202021.25


def assign_path(path: None):
    if path is not None:
        return Path(path)
    return path


class MOT16Reader(object):
    """ reader class """

    def __init__(
        self,
        root_path: str,
        depth_path: str = None,
        dets_path: str = None,
        panoptic_path: str = None,
        instance_segmentation_path: str = None,
        annotations_path: str = None,
        catgoery2class_json_path: str = None,
        metadata_path: str = None,
        correspondece_path: str = None,
        optical_flow_path: str = None,
        seq_ids: tuple = ("MOT16-03", "MOT16-01"),
        exclude_cats: list = [],
        include_cats: list = [],
        shape: list = [1080, 1920],
    ):
        """Constructor arguments

        Args:
            root_path (str): path to the MOT16 dataset
            depth_path (str, optional): path to the depth maps
            dets_path (str, optional): path to the detections
            panoptic_path (str, optional): path to the panoptic segmentation
            instance_segmentation_path (str, optional): path to segmentation masks
            annotations_path (str, optional): path to annotations files
        """
        self.root_path = Path(root_path)
        self.depth_path = assign_path(depth_path)
        self.dets_path = dets_path
        self.panoptic_path = panoptic_path
        self.instance_segmentation_path = assign_path(instance_segmentation_path)
        self.annotations_path = annotations_path
        self.catgoery2class_json_path = catgoery2class_json_path
        self.metadata_path = metadata_path
        self.seq_ids = seq_ids
        self.include_cats = include_cats
        self.exclude_cats = exclude_cats
        self.correspondece_path = assign_path(correspondece_path)
        self.shape = shape
        self.optical_flow_path = assign_path(optical_flow_path)

        self.color_ids = []
        self.category2class = {}
        self.meta_data = None
        self.color2coco = {}
        self.category_colors = {}
        self.sequence_info = {}

        self._init_metadata()
        self._init_sequence_info()
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

    def _init_sequence_info(self):
        sequence_info = {seq_id: {} for seq_id in self.seq_ids}
        for seq_id in self.seq_ids:
            parser = ConfigParser()
            config_path = self.root_path / "img1" / seq_id / "seqinfo.ini"
            imgs_path = self.root_path / "img1" / seq_id / "img1"
            parser.read(config_path)
            sequence_info[seq_id] = {
                "length": int(parser["Sequence"]["seqLength"]),
                "img_width": int(parser["Sequence"]["imWidth"]),
                "img_height": int(parser["Sequence"]["imHeight"]),
                "img_paths": sorted(imgs_path.glob("*.jpg")),
                "intrinsics": INTRINSICS[seq_id],
                "dynamic": DYNAMIC_SEQUENCES[seq_id],
            }
        self.sequence_info = sequence_info

    @property
    def categories(self):
        return list(self.category_colors.keys())

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

    def _read_human_instance_segmentation(
        self, seq_id: str, frame_id: int
    ) -> np.ndarray:
        """Reads instance segmentation masks
        Args:
            seq_id: sequence ud
            frame_id: frame id
        Returns:
            np.ndarray: float depth map
        """
        if self.instance_segmentation_path is None:
            return None

        img_path = self.sequence_info[seq_id]["img_paths"][frame_id]
        file_id = str(img_path.parts[-1].split(".")[0])
        instance_path = str(
            self.instance_segmentation_path / seq_id / str(file_id + ".png")
        )
        img = Image.open(instance_path).convert("RGBA")
        img = np.array(img)
        colors = np.unique(img.reshape(-1, img.shape[2]), axis=0)
        masks = np.zeros((colors.shape[0], img.shape[0], img.shape[1]))
        for i, color in enumerate(colors):
            ped_mask = np.all(img == color, axis=-1)
            masks[i, ...] = ped_mask
        return masks

    def _read_nbb_pts(self, seq_id: str, frame_id: int) -> tuple:
        """Reads correspondences between current frame and the next frame
        Args:
            seq_id: sequence id
            frame_id: frame id
        Returns:
            np.ndarray: [description]
        """
        if self.correspondece_path is None:
            return None, None

        crp_path = Path(self.correspondece_path) / seq_id

        def read_crp_file(filename: str) -> np.ndarray:
            file = open(filename, "r")
            lines = file.readlines()
            crp = np.zeros((len(lines), 2), dtype=np.float)
            for i, line in enumerate(lines):
                crp[i] = [int(num.strip()) for num in line.split(",")]
            return crp

        current = "correspondence_A{}_top_1000.txt".format(frame_id)
        next = current
        if frame_id + 1 < self.sequence_info[seq_id]["length"]:
            next = "correspondence_B{}_top_1000.txt".format(frame_id + 1)
        current = str(crp_path / current)
        next = str(crp_path / next)
        current, next = read_crp_file(current), read_crp_file(next)
        height = self.sequence_info[seq_id]["img_height"]
        width = self.sequence_info[seq_id]["img_width"]
        current[:, 0] *= width / 224
        current[:, 1] *= height / 224
        next[:, 0] *= width / 224
        next[:, 1] *= height / 224
        return current.astype(np.int), next.astype(np.int)

    def _read_depth(self, seq_id, frame_id) -> np.ndarray:
        """Reads depth map file
        Args:
            seq_id: sequence ud
            frame_id: frame id
        Returns:
            np.ndarray: float depth map
        """
        if self.depth_path is None:
            return None
        img_path = self.sequence_info[seq_id]["img_paths"][frame_id]
        file_id = str(img_path.parts[-1].split(".")[0])
        depth_path = str(self.depth_path / seq_id / str(file_id + ".npy"))
        depth = np.load(depth_path)
        depth = cv2.resize(
            depth, dsize=list(reversed(self.shape)), interpolation=cv2.INTER_CUBIC
        )
        depth = np.clip(depth, MIN_DEPTH, MAX_DEPTH)
        return depth

    def _read_rgb(self, seq_id, frame_id) -> np.ndarray:
        """Reads RGB image
        Args:
            seq_id: sequence ud
            frame_id: frame id
        Returns:
            np.ndarray: float depth map
        """
        # RGB path should always be present
        img_path = self.sequence_info[seq_id]["img_paths"][frame_id]
        image = utils.load_image(img_path, as_numpy=True)
        image = cv2.resize(
            image, dsize=list(reversed(self.shape)), interpolation=cv2.INTER_CUBIC
        )
        return image

    def _read_optical_flow(self, seq_id, frame_id):
        """Reads correspondences between current frame and the next frame
        Args:
            seq_id: sequence id
            frame_id: frame id
            shape: shape of the resulting optical flow array
        Returns:
            np.ndarray: optical flow between current and the next frame
        """
        if self.optical_flow_path is None:
            return None

        if frame_id == self.sequence_info[seq_id]["length"] - 1:
            return None

        if not self.sequence_info[seq_id]["dynamic"]:
            return None

        file_name = "{}_to_{}_flow.flo".format(frame_id + 1, frame_id + 2)
        file = open(str(self.optical_flow_path / seq_id / file_name), "r")
        assert np.fromfile(file, np.float32, count=1)[0] == TAG_FLOAT
        width = np.fromfile(file, np.int32, count=1)[0]
        height = np.fromfile(file, np.int32, count=1)[0]
        data = np.fromfile(file, np.float32, count=2 * width * height)
        flow = np.resize(data, (int(height), int(width), 2))
        file.close()
        flow = cv2.resize(
            flow, dsize=list(reversed(self.shape)), interpolation=cv2.INTER_CUBIC
        )
        return flow

    def read_sample(self, seq_id, frame_id):

        image = self._read_rgb(seq_id, frame_id)

        panoptic_image, panoptic_mask = self.read_panoptic_img(seq_id, frame_id)

        depth = self._read_depth(seq_id, frame_id)

        instance_masks = self._read_human_instance_segmentation(seq_id, frame_id)

        current_crp, next_crp = self._read_nbb_pts(seq_id, frame_id)

        intrinscs = INTRINSICS[seq_id].copy()

        # indexing for optical flow files starts from 1
        optical_flow = self._read_optical_flow(seq_id, frame_id)

        return {
            "image": image,
            "depth": depth,
            "panoptic_image": panoptic_image,
            "panoptic_mask": panoptic_mask,
            "intrinsics": intrinscs,
            "instance_masks": instance_masks,
            "current_correspondence": current_crp,
            "next_correspondence": next_crp,
            "optical_flow": optical_flow,
        }
