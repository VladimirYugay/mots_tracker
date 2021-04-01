#!/usr/bin/env python
import json
import logging
import os
import sys
import time

import click
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors

import mots_tracker
from mots_tracker import utils
from mots_tracker.readers import MOTSReader
from mots_tracker.trackers import MedianProjectionTracker
from mots_tracker.vis_utils import M_COLORS

_logger = logging.getLogger(__name__)


@click.command()
@click.option("--lag", default=0)
@click.option(
    "--mots_path",
    "mots_path",
    default="/home/vy/university/thesis/datasets/MOTS/",
    type=click.Path(exists=True),
    help="path to mots dataset",
)
@click.option(
    "--output_path",
    "output_path",
    default="./data/output/test",
    help="path to tracker outputs",
)
@click.option("--phase", default="train")
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
    default="./configs/tracker_configs/reprojection_tracker_config.json",
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
        axis = [fig.add_subplot(321 + i) for i in range(4)]
        traj_axis = fig.add_subplot(325, projection="3d")

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with open(str(reader_cfg_path), "r") as reader_config_file:
        reader_args = json.load(reader_config_file)
    reader = MOTSReader(os.path.join(mots_path, phase), reader_args)
    for seq in reader.sequence_info.keys():
        p_0 = np.array([0.0, 0.0, 0.0, 1.0])
        poses = np.array([0.0, 0.0, 0.0, 1.0])
        orig_width = reader.sequence_info[seq]["img_width"]
        orig_height = reader.sequence_info[seq]["img_height"]
        with open(str(tracker_cfg_path), "r") as tracker_config_file:
            tracker_args = json.load(tracker_config_file)
        mot_tracker = MedianProjectionTracker(*tracker_args.values())
        out_file = open(os.path.join(output_path, "{}.txt".format(seq)), "w")
        logging.log(log_level, "Processing %s." % seq)
        for frame in range(reader.sequence_info[seq]["length"]):
            logging.log(log_level, "Processing frame {}".format(frame + 1))
            sample = reader.read_sample(seq, frame)
            frame += 1

            if display:
                axis[0].set_title("Sequence: {}, frame: {}".format(seq, frame))
                axis[0].imshow(sample["image"])
                axis[1].set_title("GT boxes")
                axis[1].imshow(sample["image"])
                for box_id, bb in enumerate(sample["boxes"]):
                    bb = bb.astype(np.int32)
                    axis[1].add_patch(
                        patches.Rectangle(
                            (bb[0], bb[1]),
                            bb[2] - bb[0],
                            bb[3] - bb[1],
                            fill=False,
                            lw=3,
                            color=M_COLORS[box_id],
                        )
                    )

                axis[2].set_title("Projections")
                display_img = sample["image"].copy()
                for proj_id, proj in enumerate(mot_tracker.projections):
                    color = np.asarray(
                        np.asarray(mcolors.to_rgb(M_COLORS[proj_id])) * 255,
                        dtype=int,
                    )
                    display_img[proj == 1] = color
                axis[2].imshow(display_img)

                axis[3].set_title("GT masks")
                display_img = sample["image"].copy()
                for mask_id, mask in enumerate(sample["masks"]):
                    color = np.asarray(
                        np.asarray(mcolors.to_rgb(M_COLORS[mask_id])) * 255,
                        dtype=int,
                    )
                    display_img[mask == 1] = color
                axis[3].imshow(display_img)

                p_acc = mot_tracker.accumulated_egomotion.dot(p_0)
                poses = np.vstack((poses, p_acc))
                traj_axis.scatter(
                    poses[:, 0], poses[:, 1], poses[:, 2], c=np.arange(poses.shape[0])
                )

            trackers = mot_tracker.update(sample, sample["intrinsics"])

            for (_, _, box, idx) in trackers:
                state = utils.resize_boxes(
                    box[None, :], (416, 128), (orig_width, orig_height)
                )[0]
                # state = box
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
                    box = box.astype(np.int32)
                    axis[0].add_patch(
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
                time.sleep(lag)
                fig.canvas.flush_events()
                plt.draw()
                for ax in axis:
                    ax.cla()
                traj_axis.cla()

        out_file.close()


if __name__ == "__main__":
    main()
