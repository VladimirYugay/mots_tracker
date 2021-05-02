#!/usr/bin/env python
import json
import logging
import os
import sys
import time
from multiprocessing import Pool

import click
import matplotlib.pyplot as plt
import numpy as np

import mots_tracker
from mots_tracker import io_utils, utils, vis_utils
from mots_tracker.readers import MOTSynthReader
from mots_tracker.trackers import BBox3dTracker

_logger = logging.getLogger(__name__)


@click.command()
@click.option("--c", "--cores", "cores", default=4)
@click.option(
    "--dp",
    "--data_path",
    "data_path",
    default="/home/vy/university/thesis/datasets/MOTS/",
    type=click.Path(exists=True),
    help="Path to the dataset",
)
@click.option("--lag", default=0)
@click.option(
    "--op",
    "--output_path",
    "output_path",
    default="./data/output/test",
    help="path to tracker outputs",
)
@click.option("--phase", default="train")
@click.option(
    "--rc",
    "--reader_config",
    "reader_cfg_path",
    default="./configs/reader_configs/motsynth_reader_config.json",
    type=click.Path(exists=True),
    help="path to reader config file",
)
@click.option(
    "--tc",
    "--tracker_config",
    "tracker_cfg_path",
    default="./configs/tracker_configs/bbox3d_tracker_config.json",
    type=click.Path(exists=True),
    help="path to tracker config file",
)
@click.option(
    "--display",
    "display",
    help="Display online tracker mots_output (slow) [False]",
    flag_value=True,
)
@click.option("-v", "--verbose", "log_level", flag_value=logging.INFO, default=True)
@click.version_option(mots_tracker.__version__)
def main(
    cores,
    data_path,
    output_path,
    display,
    lag,
    phase,
    reader_cfg_path,
    tracker_cfg_path,
    log_level,
):
    """
    mot_path: Path to the dataset \n
    output_path: Path to the output folder \n
    lag: Time to wait during display \n
    reader_config: Path to the reader configuration \n
    tracker_config: Path to the tracker configuration \n
    phase: Training of validation phase
    """
    logging.basicConfig(
        stream=sys.stdout,
        level=log_level,
        datefmt="%Y-%m-%d %H:%M",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    with open(str(reader_cfg_path), "r") as reader_config_file:
        reader_config = json.load(reader_config_file)
    with open(str(tracker_cfg_path), "r") as tracker_config_file:
        tracker_config = json.load(tracker_config_file)
    reader = MOTSynthReader(os.path.join(data_path, phase), reader_config)
    seq_ids = sorted(reader.sequence_info.keys())

    if not display:
        pool = Pool(cores)
        mot_tracker = BBox3dTracker(*tracker_config.values())  # each for one sequence
        args = [
            (reader, seq_id, output_path, mot_tracker, reader_config)
            for seq_id in seq_ids
        ]
        pool.map(io_utils.multi_run_wrapper, args)
        return

    plt.ion()
    fig = plt.figure()
    axis_track = fig.add_subplot(121, aspect="equal")
    axis_gt = fig.add_subplot(122, aspect="equal")

    for seq in seq_ids:
        mot_tracker = BBox3dTracker(*tracker_config.values())
        out_file = open(os.path.join(output_path, "{}.txt".format(seq)), "w")
        logging.log(log_level, "Processing %s." % seq)
        for frame in range(reader.sequence_info[seq]["length"]):
            sample = reader.read_sample(seq, frame)
            frame += 1
            logging.log(log_level, "Processing frame {}".format(frame))

            axis_track.imshow(sample["image"])
            axis_track.set_title("Sequence: {}, frame: {}".format(seq, frame))
            axis_gt.set_title("Ground truth")
            axis_gt.imshow(sample["image"])

            for box_id, bb in zip(sample["box_ids"], sample["boxes"]):
                bb = bb.astype(np.int32)
                vis_utils.plot_box_patch(axis_gt, bb, box_id)

            trackers = mot_tracker.update(sample, sample["intrinsics"])

            for (_, _, box, idx) in trackers:
                if display:
                    vis_utils.plot_box_patch(axis_track, box, idx)
                if "resize_shape" in reader_config and reader_config["resize_shape"]:
                    width = reader.sequence_info[seq]["img_width"]
                    height = reader.sequence_info[seq]["img_height"]
                    box = utils.resize_boxes(
                        box, reader_config["resize_shape"], (width, height)
                    )[0]
                io_utils.print_mot_format(frame, idx, box, out_file)
            time.sleep(lag)
            fig.canvas.flush_events()
            plt.draw()
            axis_track.cla()
            axis_gt.cla()

        out_file.close()


if __name__ == "__main__":
    main()
