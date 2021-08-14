""" Generates motsynth segmentation images  """
import logging
import sys
from multiprocessing import Pool
from pathlib import Path

import click

from mots_tracker import readers, utils
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
    (output_path / seq_id).mkdir(parents=True, exist_ok=True)
    for frame_id in range(reader.sequence_info[seq_id]["length"]):
        logging.log(
            logging.INFO, "Processing sequence: {}, frame: {}".format(seq_id, frame_id)
        )
        sample = reader.read_sample(seq_id, frame_id)
        seg_img = sample["masks"].sum(axis=0)
        file_name = "{0:04d}".format(frame_id) + ".png"
        utils.save_img(seg_img, str((output_path / seq_id / file_name)))


@click.command()
@click.option("--c", "--cores", "cores", default=4)
@click.option(
    "--op",
    "--output_path",
    "output_path",
    default="/home/vy/university/thesis/datasets/MOTSynth",
    type=click.Path(exists=True),
    help="Path to resulting segmentation masks",
)
@click.option(
    "--cp",
    "--config_path",
    "config_path",
    default="./configs/median_tracker_config.yaml",
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
    output_path = output_path / "seg_frames"
    output_path.mkdir(parents=True, exist_ok=True)

    config = load_yaml(config_path)
    reader = get_instance(readers, "reader", config)
    seq_ids = sorted(reader.sequence_info.keys())
    pool = Pool(cores)
    args = [(reader, seq_id, output_path) for seq_id in seq_ids]
    pool.map(multi_run_wrapper, args)


if __name__ == "__main__":
    main()
