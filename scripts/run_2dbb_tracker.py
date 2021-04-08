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
from mots_tracker import readers, utils
from mots_tracker.io_utils import print_mot_format

# from mots_tracker.io_utils import print_kitti_format
from mots_tracker.trackers import BBox2dTracker
from mots_tracker.vis_utils import M_COLORS

_logger = logging.getLogger(__name__)


@click.command()
@click.option("--lag", default=0)
@click.option(
    "--dp",
    "--data_path",
    "mots_path",
    default="/home/vy/university/thesis/datasets/MOTS/",
    type=click.Path(exists=True),
    help="Path to the dataset: MOTS, MOTSynth, KITTI",
)
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
    "--reader_cfg_path",
    "reader_cfg_path",
    default="./configs/reader_configs/mots_reader_config.json",
    type=click.Path(exists=True),
    help="path to reader config file",
)
@click.option(
    "--tc",
    "--tracker_cfg_path",
    "tracker_cfg_path",
    default="./configs/tracker_configs/bbox2d_tracker_config.json",
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
        reader_config = json.load(reader_config_file)
    reader = readers.MOTSReader(os.path.join(mots_path, phase), reader_config)
    for seq in sorted(reader.sequence_info.keys()):
        with open(str(tracker_cfg_path), "r") as tracker_config_file:
            tracker_args = json.load(tracker_config_file)
        mot_tracker = BBox2dTracker(*tracker_args.values())
        out_file = open(os.path.join(output_path, "{}.txt".format(seq)), "w")
        for frame in range(reader.sequence_info[seq]["length"]):
            logging.log(
                log_level, "Processing sequence {}, frame {}".format(seq, frame + 1)
            )
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

            trackers = mot_tracker.update(sample, sample["intrinsics"])

            for (pred_box, box, idx, obj_type) in trackers:
                if "resize_shape" in reader_config:
                    width = reader.sequence_info[seq]["img_width"]
                    height = reader.sequence_info[seq]["img_height"]
                    box = utils.resize_boxes(
                        box[None, :], reader_config["resize_shape"], (width, height)
                    )
                print_mot_format(frame, idx, box, out_file)
                # print_kitti_format(frame - 1, idx, obj_type, box, out_file)
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
