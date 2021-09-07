""" Generates motsynth segmentation images  """
import logging
import sys
from multiprocessing import Pool
from pathlib import Path

import click
import cv2
import numpy as np

from mots_tracker import readers
from mots_tracker.io_utils import get_instance, load_yaml

_logger = logging.getLogger(__name__)


def multi_run_wrapper(args):
    """ Unpacks argument for running on multiple cores """
    return make_masks(*args)


def make_masks(reader, seq_id, output_path):
    """Function to run trackers on multiple cores
    Args:
        reader (Reader): one of the readers implemented in readers module
        seq_id (str): id of the sequence
        output_path (Path): path to save output
    """
    save_path = output_path / seq_id / "seg_frames"
    save_path.mkdir(parents=True, exist_ok=True)
    for frame_id in range(reader.sequence_info[seq_id]["length"]):
        logging.log(
            logging.INFO, "Processing sequence: {}, frame: {}".format(seq_id, frame_id)
        )
        sample = reader.read_sample(seq_id, frame_id)
        masks = sample["masks"]
        instance_seg_mask = []
        for color, mask in enumerate(masks):
            mask[mask == 1] = color + 1
            instance_seg_mask.append(mask)
        instance_seg_mask = np.array(instance_seg_mask).sum(axis=0)
        file_name = "{0:04d}".format(frame_id) + ".png"
        cv2.imwrite(str(save_path / file_name), instance_seg_mask)


@click.command()
@click.option("--c", "--cores", "cores", default=4)
@click.option(
    "--op",
    "--output_path",
    "output_path",
    default="/home/vy/university/thesis/datasets/MOTS/train/",
    type=click.Path(exists=True),
    help="Path to resulting segmentation masks",
)
@click.option(
    "--cp",
    "--config_path",
    "config_path",
    default="./configs/3dbb_tracker_config.yaml",
    type=click.Path(exists=True),
    help="path to the config file",
)
def main(output_path, config_path, cores):
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    config = load_yaml(config_path)
    reader = get_instance(readers, "reader", config)
    seq_ids = sorted(reader.sequence_info.keys())
    pool = Pool(cores)
    args = [(reader, seq_id, output_path) for seq_id in seq_ids]
    pool.map(multi_run_wrapper, args)


if __name__ == "__main__":
    main()
