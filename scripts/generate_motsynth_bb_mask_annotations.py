""" Generates masks and bb ground truth for MOT challenge from MOTSynth
    It's assumed to read data in the format on the server
    in /storage/user/brasoand/motsyn2 on 13.03.2021
    The reason to generate them together is high amount of bounding boxes
    without pedestrian rendered inside, while masks are consistent in that
    case i.e. empty. So we take only those with masks != 0
"""
import json
import logging
import sys
from configparser import ConfigParser
from pathlib import Path

import click

from mots_tracker.utils import decode_mask

_logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--input_path",
    "input_path",
    default="/home/vy/university/thesis/datasets/MOTSynth/annotations",
    type=click.Path(exists=True),
    help="Path to raw annotations",
)
@click.option(
    "--output_path",
    "output_path",
    default="/home/vy/university/thesis/datasets/MOTSynth",
    type=click.Path(exists=True),
    help="Path to resulting gt annotations",
)
def main(input_path, output_path):
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    input_path, output_path = Path(input_path), Path(output_path)
    logging.log(logging.INFO, "Start generating bb gt")
    for ann_name in input_path.glob("**/*"):
        logging.log(
            logging.INFO, "Generating bb gt for frame: {}".format(str(ann_name))
        )
        (output_path / "bb_annotations").mkdir(parents=True, exist_ok=True)
        (output_path / "mask_annotations").mkdir(parents=True, exist_ok=True)
        (output_path / "sequences_info").mkdir(parents=True, exist_ok=True)
        with open(str(ann_name), "r") as ann_file:
            annotations = json.load(ann_file)
            # generate sequence info folder
            info_file_name = (str(ann_name).split("/")[-1]).split(".")[0] + ".ini"
            config = ConfigParser()
            config.optionxform = str
            config.add_section("Sequence")
            config.set("Sequence", "name", info_file_name.split(".")[0])
            config.set("Sequence", "seqLength", str(len(annotations["images"])))
            config.set("Sequence", "imWidth", str(annotations["images"][0]["width"]))
            config.set("Sequence", "imHeight", str(annotations["images"][0]["height"]))
            with open(
                str(output_path / "sequences_info" / info_file_name), "w"
            ) as info_output_file:
                config.write(info_output_file)

            file_name = (str(ann_name).split("/")[-1]).split(".")[0] + ".txt"
            bb_output_file = open(str(output_path / "bb_annotations" / file_name), "w")
            mask_output_file = open(
                str(output_path / "mask_annotations" / file_name), "w"
            )
            for annotation in annotations["annotations"]:
                frame_id = annotation["frame_n"]
                person_id = annotation["ped_id"]
                box = annotation["bbox"]
                height, width = annotation["segmentation"]["size"]
                mask_string = annotation["segmentation"]["counts"]
                mask = decode_mask(height, width, mask_string)
                if mask.sum() == 0:  # empty mask
                    continue
                logging.log(
                    logging.INFO,
                    "Processing sequence: {}, frame: {}".format(ann_name, frame_id + 1),
                )
                print(
                    "%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1"
                    % (frame_id + 1, person_id, box[0], box[1], box[2], box[3]),
                    file=bb_output_file,
                )
                print(
                    "{} {} {} {} {} {}".format(
                        frame_id + 1, person_id, 2, height, width, mask_string
                    ),
                    file=mask_output_file,
                )
    logging.log(logging.INFO, "Saved to: {}".format(output_path))


if __name__ == "__main__":
    main()
