""" Utils for input and output of tracker data """
from pathlib import Path

import yaml

from mots_tracker import utils


def print_mot_format(frame_id, obj_id, box, file):
    """Prints results in mot format to the file
    Args:
        frame_id (int): frame id, 1 indexed for MOT
        obj_id (int): id of the object
        box (ndarray): bounding box in (x1, y1, x2, y2) format
        file (File): opened file to write to  (do not close)
    """
    print(
        "%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1"
        % (
            frame_id,
            obj_id,
            box[0],
            box[1],
            box[2] - box[0],
            box[3] - box[1],
        ),
        file=file,
    )


def print_mots_format(frame_id: int, obj_id: int, height: int, width: int, mask, file):
    """Prints results in MOTS format to the file
    Args:
        frame_id: frame id, 1 indexed for MOT
        obj_id: id of the object
        height: height of the image
        width: width of the image
        mask: rle string
        file: opened file to write to  (do not close)
    """
    print(
        "{} {} 2 {} {} {}".format(frame_id, obj_id, height, width, mask),
        file=file,
    )


def print_kitti_format(frame_id, obj_id, obj_type, box, file):
    """Prints results in mot format to the file
    Args:
        frame_id (int): frame id, 0 indexed for KITTI
        obj_id (int): id of the object
        obj_type (str): type of the object
        box (ndarray): bounding box in (x1, y1, x2, y2) format
        file (File): opened file to write to  (do not close)
    """
    print(
        "{} {} {} -1 -1 -1 {} {} {} {} -1 -1 -1 -1 -1 -1 -1 -1".format(
            frame_id - 1, obj_id, obj_type, box[0], box[1], box[2], box[3]
        ),
        file=file,
    )


def get_instance(module, name: str, cfg: dict):
    """Instantiates a class from configuration file
    Args:
        module: python module where a class is implemented
        name: name of part of the config to instantiate
        cfg: config with parameters for the object instantiation
    Returns:
        an instance of a class defined in module and the config
    """
    return getattr(module, cfg[name]["type"])(**cfg[name]["args"])


def load_yaml(path: str) -> dict:
    """Reads yaml file
    Args:
        path: path the yaml file
    Returns:
        obj: loaded yaml object
    """
    with open(path, "r") as yaml_file:
        obj = yaml.safe_load(yaml_file)
    return obj


def multi_run_wrapper(args):
    """ Unpacks argument for running on multiple cores """
    return track_objects(*args)


def track_objects(
    reader, seq_id: str, output_path: str, mot_tracker, reader_config: dict
):
    """Function to run trackers on multiple cores
    Args:
        reader (Reader): one of the readers implemented in readers module
        seq_id: id of the sequence
        output_path: path to save output
        mot_tracker (Tracker): one of the trackers implemented in trackers module
        reader_config: configuration of the reader
    """
    # we will write both masks and boxes
    output_path = Path(output_path)
    (output_path / "MOT").mkdir(parents=True, exist_ok=True)
    (output_path / "MOTS").mkdir(parents=True, exist_ok=True)
    mot_out_file = open(str(output_path / "MOT" / "{}.txt".format(seq_id)), "w")
    mots_out_file = open(str(output_path / "MOTS" / "{}.txt".format(seq_id)), "w")
    width = reader.sequence_info[seq_id]["img_width"]
    height = reader.sequence_info[seq_id]["img_height"]
    print("Processing %s." % seq_id)
    for frame in range(reader.sequence_info[seq_id]["length"]):
        print("Processing seq: {}, frame: {}".format(seq_id, frame))
        sample = reader.read_sample(seq_id, frame)
        frame += 1
        trackers = mot_tracker.update(sample, sample["intrinsics"])
        for (
            raw_mask,
            box,
            idx,
        ) in trackers:  # we don't use predicted box for our experiments
            if "resize_shape" in reader_config and reader_config["resize_shape"]:
                box = utils.resize_boxes(
                    box[None, :], reader_config["resize_shape"], (width, height)
                )[0]
            print_mot_format(frame, idx, box, mot_out_file)
            print_mots_format(frame, idx, height, width, raw_mask, mots_out_file)
    mot_out_file.close()
    mots_out_file.close()
