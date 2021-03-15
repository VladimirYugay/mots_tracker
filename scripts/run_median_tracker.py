#!/usr/bin/env python
import json
import logging
import os
import sys

import click
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

import mots_tracker
from mots_tracker import utils
from mots_tracker.readers import MOTSReader
from mots_tracker.trackers import MedianTracker
from mots_tracker.vis_utils import M_COLORS

_logger = logging.getLogger(__name__)


@click.command()
@click.argument("lag", default=0)
@click.argument("mots_path", default="/home/vy/university/thesis/datasets/MOTS/")
@click.argument("output_path", default="./data/output/test")
@click.argument("phase", default="train")
@click.option(
    "-rc",
    "--reader_config",
    "reader_cfg_path",
    default="./configs/reader_configs/mots_reader_config.json",
    type=click.Path(exists=True),
    help="path to reader config file",
)
@click.option(
    "-tc",
    "--tracker_config",
    "tracker_cfg_path",
    default="./configs/tracker_configs/median_tracker_config.json",
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
    mots_path,
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

    if display:
        plt.ion()
        fig = plt.figure()
        axis_track = fig.add_subplot(121, aspect="equal")
        axis_gt = fig.add_subplot(122, aspect="equal")

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with open(str(reader_cfg_path), "r") as reader_config_file:
        reader_args = json.load(reader_config_file)
    reader = MOTSReader(os.path.join(mots_path, phase), reader_args)
    for seq in reader.sequence_info.keys():
        orig_width = reader.sequence_info[seq]["img_width"]
        orig_height = reader.sequence_info[seq]["img_height"]
        intrinsics = reader.sequence_info[seq]["intrinsics"]
        with open(str(tracker_cfg_path), "r") as tracker_config_file:
            tracker_args = json.load(tracker_config_file)
        mot_tracker = MedianTracker(*tracker_args.values())
        out_file = open(os.path.join(output_path, "{}.txt".format(seq)), "w")
        logging.log(log_level, "Processing %s." % seq)
        for frame in range(reader.sequence_info[seq]["length"]):
            logging.log(log_level, "Processing frame {}".format(frame + 1))
            sample = reader.read_sample(seq, frame)
            frame += 1

            if display:
                axis_track.imshow(sample["image"])
                axis_track.set_title("Sequence: {}, frame: {}".format(seq, frame))
                axis_gt.set_title("Ground truth")
                axis_gt.imshow(sample["image"])

                for box_id, bb in enumerate(sample["boxes"]):
                    bb = bb.astype(np.int32)
                    axis_gt.add_patch(
                        patches.Rectangle(
                            (bb[0], bb[1]),
                            bb[2] - bb[0],
                            bb[3] - bb[1],
                            fill=False,
                            lw=3,
                            color=M_COLORS[box_id],
                        )
                    )

            trackers = mot_tracker.update(sample, intrinsics)

            for (_, _, box, idx) in trackers:
                state = utils.resize_boxes(
                    box[None, :], (416, 128), (orig_width, orig_height)
                )[0]
                print(
                    "%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1"
                    % (
                        frame,
                        idx,
                        state[0],
                        state[1],
                        state[2] - state[0],
                        state[3] - state[1],
                    ),
                    file=out_file,
                )
                if display:
                    axis_track.add_patch(
                        patches.Rectangle(
                            (box[0], box[1]),
                            box[2] - box[0],
                            box[3] - box[1],
                            fill=False,
                            lw=3,
                            color=M_COLORS[idx],
                        )
                    )
            if display:
                import time

                time.sleep(lag)
                fig.canvas.flush_events()
                plt.draw()
                axis_track.cla()
                axis_gt.cla()

        out_file.close()


if __name__ == "__main__":
    main()
