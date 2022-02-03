""" module for reading mots ground truth data """
import itertools
import json
import pickle
from argparse import Namespace
from configparser import ConfigParser
from pathlib import Path

import numpy as np

from mots_tracker import utils

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


class MOT16Reader(object):
    """ reader class """

    def __init__(
        self,
        root_path: str,
        depth_path: str = None,
        dets_path: str = None,
        panoptic_path: str = None,
        segmentation_path: str = None,
        annotations_path: str = None,
        catgoery2class_json_path: str = None,
        metadata_path: str = None,
        seq_ids: tuple = ("MOT16-03", "MOT16-01"),
    ):
        """Constructor arguments

        Args:
            root_path (str): path to the MOT16 dataset
            depth_path (str, optional): path to the depth maps
            dets_path (str, optional): path to the detections
            panoptic_path (str, optional): path to the panoptic segmentation
            segmentation_path (str, optional): path to segmentation masks
            annotations_path (str, optional): path to annotations files
        """
        self.root_path = Path(root_path)
        self.depth_path = depth_path
        self.dets_path = dets_path
        self.panoptic_path = Path(panoptic_path)
        self.segmentation_path = segmentation_path
        self.annotations_path = annotations_path
        self.catgoery2class_json_path = catgoery2class_json_path
        self.metadata_path = metadata_path
        self.seq_ids = seq_ids

        self.color_ids = []
        self.category2class = {}
        self.metadata = None
        self.color2coco = {}
        self.category_colors = {}
        self.sequence_info = {}

        self._init_sequence_info()
        self._init_color_ids()
        self._init_category2class()
        self._init_metadata()
        # self._init_color2coco()
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
        """ Intializes metadata on the panoptic classes """
        with open(self.metadata_path, "rb") as pkl_file:
            self.meta_data = Namespace(**pickle.load(pkl_file))
        print(type(self.meta_data))

    def _init_color2coco(self):
        """ Initializes mapping of color to coco """
        self.color_to_coco = {}
        for stuff_class, stuff_color in zip(
            self.metadata.stuff_classes, self.metadata.stuff_colors
        ):
            color = " ".join(str(e) for e in stuff_color)
            self.statement[color] = stuff_class
        for thing_class, thing_color in zip(
            self.metadata.thing_classes, self.metadata.thing_colors
        ):
            color = " ".join(str(e) for e in thing_color)
            self.color_to_coco[color] = thing_class

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
            }
        print(sequence_info.keys())
        self.sequence_info = sequence_info

    @property
    def categories(self):
        return list(self.category_colors.keys())

    def read_panoptic_img(
        self, img_filename: str, exclude_cats: list = [], include_cats: list = []
    ):
        """Reads panoptic image
        Args:
            img_filename (str): path to the panoptic image
            exclude_cats (list, optional): categories to exclude
            include_cats (list, optional): categories to include
        """
        img = utils.load_image(img_filename)
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
            if include_cats and category not in include_cats:
                continue
            if exclude_cats and category in exclude_cats:
                continue
            final_color = self.category_colors[category]
            color_indices = np.where(np.all(img == color, axis=-1))
            panoptic_img[color_indices] = final_color
            panoptic_mask[color_indices] = self.categories.index(category)
        return panoptic_img, panoptic_mask

    def read_sample(self, seq_id, frame_id):
        img_path = self.sequence_info[seq_id]["img_paths"][frame_id]
        image = utils.load_image(img_path)
        panoptic_image, panoptic_mask = self.read_panoptic_img(
            str(self.panoptic_path / seq_id / str(img_path.parts[-1]).replace(".jpg", ".png"))
        )
        return {
            "image": image,
            "panoptic_image": panoptic_image,
            "panoptic_mask": panoptic_mask,
        }
