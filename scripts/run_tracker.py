#!/usr/bin/env python
import logging
import sys
from multiprocessing import Pool
from pathlib import Path

import click
import yaml

import mots_tracker
from mots_tracker import readers, trackers
from mots_tracker.io_utils import get_instance, multi_run_wrapper

_logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--cp",
    "--config_path",
    "config_path",
    default="/home/vy/university/thesis/datasets/MOTSynth/",
    type=click.Path(exists=True),
    help="Path to the dataset: MOTS, MOTSynth, KITTI",
)
@click.option("-v", "--verbose", "log_level", flag_value=logging.INFO, default=True)
@click.version_option(mots_tracker.__version__)
def main(config_path, log_level):
    """
    config_path: Path to the dataset \n
    log_level: Logging level \n
    """
    logging.basicConfig(
        stream=sys.stdout,
        level=log_level,
        datefmt="%Y-%m-%d %H:%M",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)

    Path(config["output_path"]).mkdir(parents=True, exist_ok=True)
    reader = get_instance(readers, "reader", config)
    seq_ids = sorted(reader.sequence_info.keys())

    pool = Pool(config["cores"])
    tracker = get_instance(trackers, "tracker", config)
    args = [
        (reader, seq_id, config["output_path"], tracker, config["tracker"]["args"])
        for seq_id in seq_ids
    ]
    pool.map(multi_run_wrapper, args)


if __name__ == "__main__":
    main()
