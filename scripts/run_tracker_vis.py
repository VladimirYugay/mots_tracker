#!/usr/bin/env python
import logging
import sys
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import yaml

import mots_tracker
from mots_tracker import readers, trackers, utils, vis_utils
from mots_tracker.io_utils import get_instance, print_mot_format, print_mots_format

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

    reader_config = config["reader"]["args"]
    output_path = Path(config["output_path"])
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "MOTS").mkdir(parents=True, exist_ok=True)
    (output_path / "MOT").mkdir(parents=True, exist_ok=True)
    reader = get_instance(readers, "reader", config)
    seq_ids = sorted(reader.sequence_info.keys())

    plt.ion()
    fig = plt.figure()
    axis_track = fig.add_subplot(121, aspect="equal")
    axis_gt = fig.add_subplot(122, aspect="equal")

    for seq in seq_ids:
        tracker = get_instance(trackers, "tracker", config)
        width = reader.sequence_info[seq]["img_width"]
        height = reader.sequence_info[seq]["img_height"]
        mot_out_file = open(str(output_path / "MOT" / "{}.txt".format(seq)), "w")
        mots_out_file = open(str(output_path / "MOTS" / "{}.txt".format(seq)), "w")
        for frame in range(reader.sequence_info[seq]["length"]):
            logging.log(
                log_level, "Processing sequence {}, frame {}".format(seq, frame + 1)
            )
            sample = reader.read_sample(seq, frame)
            frame += 1

            axis_track.imshow(sample["image"])
            axis_track.set_title("Sequence: {}, frame: {}".format(seq, frame))
            axis_gt.set_title("Ground truth")
            axis_gt.imshow(sample["image"])

            for box_id, bb in enumerate(sample["boxes"]):
                bb = bb.astype(np.int32)
                vis_utils.plot_box_patch(axis_gt, bb, box_id)

            tracks = tracker.update(sample, sample["intrinsics"])

            for (raw_mask, box, idx) in tracks:
                if (
                    "resize_shape" in reader_config
                    and reader_config["resize_shape"] is not None
                ):
                    box = utils.resize_boxes(
                        box[None, :], reader_config["resize_shape"], (width, height)
                    )
                print_mot_format(frame, idx, box, mot_out_file)
                print_mots_format(frame, idx, height, width, raw_mask, mots_out_file)
                vis_utils.plot_box_patch(axis_track, box, idx)

            fig.canvas.flush_events()
            plt.draw()
            axis_track.cla()
            axis_gt.cla()

        mot_out_file.close()
        mots_out_file.close()


if __name__ == "__main__":
    main()
