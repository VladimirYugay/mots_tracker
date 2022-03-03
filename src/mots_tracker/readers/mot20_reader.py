""" module for reading mots ground truth data """
import itertools
import json
import pickle
from argparse import Namespace
from configparser import ConfigParser
from pathlib import Path

import cv2
import numpy as np

from mots_tracker import utils

MIN_DEPTH = 0.1  # in meters
MAX_DEPTH = 1e6  # in meters

INTRINSICS = np.array([[1539.579, 0, 940.66], [0, 1542.185, 556.894], [0, 0, 1]])

SEQ_SIZES = {
    "MOT20-01": [1080, 1920],
    "MOT20-02": [1080, 1920],
    "MOT20-03": [880, 1173],
    "MOT20-04": [1080, 1545],
    "MOT20-05": [1080, 1654],
    "MOT20-06": [734, 1920],
    "MOT20-07": [1080, 1920],
    "MOT20-08": [734, 1920],
}

SEQ_SPLIT = {
    "MOT20-01": "train",
    "MOT20-02": "train",
    "MOT20-03": "train",
    "MOT20-04": "test",
    "MOT20-05": "train",
    "MOT20-06": "test",
    "MOT20-07": "test",
    "MOT20-08": "test",
}


def assign_path(path: None):
    if path is not None:
        return Path(path)
    return path


class MOT20Reader(object):
    """ reader class """

    def __init__(
        self,
        root_path: str,
        catgoery2class_json_path: str = None,
        metadata_path: str = None,
        seq_ids: tuple = ("MOT16-04", "MOT16-06"),
        exclude_cats: list = [],
        include_cats: list = [],
    ):
        """Constructor arguments

        Args:
            root_path (str): path to the MOT16 dataset
            depth_path (str, optional): path to the depth maps
            annotations_path (str, optional): path to annotations files
        """
        self.root_path = Path(root_path)
        self.catgoery2class_json_path = catgoery2class_json_path
        self.metadata_path = metadata_path
        self.seq_ids = seq_ids
        self.include_cats = include_cats
        self.exclude_cats = exclude_cats

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
            path = self.root_path / SEQ_SPLIT[seq_id] / seq_id
            config_path = path / "seqinfo.ini"
            imgs_path = path / "img1"
            parser.read(config_path)
            sequence_info[seq_id] = {
                "length": int(parser["Sequence"]["seqLength"]),
                "img_width": int(parser["Sequence"]["imWidth"]),
                "img_height": int(parser["Sequence"]["imHeight"]),
                "img_paths": sorted(imgs_path.glob("*.jpg")),
                "intrinsics": INTRINSICS,
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
        img_path = self.sequence_info[seq_id]["img_paths"][frame_id]
        file_id = str(img_path.parts[-1]).replace(".jpg", ".png")
        panoptic_path = Path(*img_path.parts[:-2]) / "panoptic" / seq_id / file_id
        img = utils.load_image(panoptic_path, as_numpy=True)
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

    def _read_depth(self, seq_id, frame_id) -> np.ndarray:
        """Reads depth map file
        Args:
            seq_id: sequence ud
            frame_id: frame id
        Returns:
            np.ndarray: float depth map
        """
        img_path = self.sequence_info[seq_id]["img_paths"][frame_id]
        file_id = str(img_path.parts[-1].split(".")[0])
        depth_path = Path(*img_path.parts[:-2]) / "depth" / str(file_id + ".npz")
        depth = np.load(depth_path)["arr_0"]
        depth = cv2.resize(
            depth,
            dsize=list(reversed(SEQ_SIZES[seq_id])),
            interpolation=cv2.INTER_CUBIC,
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
        img_path = self.sequence_info[seq_id]["img_paths"][frame_id]
        image = utils.load_image(img_path, as_numpy=True)
        return image

    def read_sample(self, seq_id, frame_id):
        image = self._read_rgb(seq_id, frame_id)
        panoptic_image, panoptic_mask = self.read_panoptic_img(seq_id, frame_id)
        depth = self._read_depth(seq_id, frame_id)
        intrinscs = INTRINSICS.copy()
        return {
            "image": image,
            "depth": depth,
            "panoptic_image": panoptic_image,
            "panoptic_mask": panoptic_mask,
            "intrinsics": intrinscs,
        }
