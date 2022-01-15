""" Generates 2D and 3D keypoints ground truth for MOT challenge from MOTSynth
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
from multiprocessing import Pool
from pathlib import Path

import click
import numpy as np
from pycocotools import mask as rletools

_logger = logging.getLogger(__name__)


def multi_run_wrapper(args):
    return generate_mot_file(*args)


def generate_mot_file(ann_name, output_path):
    logging.log(logging.INFO, "Generating gt keypoints for frame: {}".format(str(ann_name)))
    with open(str(ann_name), "r") as ann_file:
        annotations = json.load(ann_file)
        seq_name = ann_name.parts[-1].split(".")[0]
        output_path = output_path / seq_name
        (output_path / "gt").mkdir(parents=True, exist_ok=True)
        # generate sequence info folder
        create_info_file(annotations, output_path)
        # generate bounding boxes and masks annotations
        keypoints_3d_output_file = open(str(output_path / "gt" / "keypoints_3d.txt"), "w")
        keypoints_2d_output_file = open(str(output_path / "gt" / "keypoints_2d.txt"), "w")
        for annotation in annotations["annotations"]:
            if ann_name.parts[-1] == "000.json":
                frame_id = annotation["image_id"]
            else:
                frame_id = int(
                    str(annotation["image_id"])[-4:]
                )  # temporal fix due to ann encoding

            visibility = np.array(annotation["keypoints"]).reshape(-1, 3)[:, -1]
            if np.all(
                visibility == 1
            ):  # filter out objects with all joints occluded
                # 1 - invisible, 2 - visible
                continue

            person_id = annotation["ped_id"]
            height, width = annotation["segmentation"]["size"]
            mask_string = annotation["segmentation"]["counts"]

            if rletools.area(annotation["segmentation"]) < 800:
                continue
            
            keypoints_2d_line = ','.join(map(str, 
                [frame_id + 1, person_id] + annotation["keypoints"]))
            keypoints_3d_line = ','.join(map(str, 
                [frame_id + 1, person_id] + annotation["keypoints_3d"]))
            print(keypoints_2d_line, file=keypoints_2d_output_file)
            print(keypoints_3d_line, file=keypoints_3d_output_file)            
        keypoints_2d_output_file.close()
        keypoints_3d_output_file.close()

        # generate egomotion annotations
        create_egomotion_file(annotations, output_path)


def create_egomotion_file(ann, output_path):
    """ Creates file with camera egomotion """
    egomotion_output_file = open(str(output_path / "gt" / "egomotion.txt"), "w")
    for img_ann in ann["images"]:
        rotation = img_ann["cam_world_rot"]
        translation = img_ann["cam_world_pos"]
        fov = img_ann["cam_fov"]
        print(
            "{} {} {} {} {} {} {}".format(*rotation, *translation, fov),
            file=egomotion_output_file,
        )
    egomotion_output_file.close()


def create_info_file(ann, output_path):
    """ Creates sequence info file from annotation """

    def is_moving(img_anns):
        """ If camera is moving in a given sequence """
        for img_ann in img_anns:
            if img_ann["is_moving"]:
                return 1
        return 0

    config = ConfigParser()
    config.optionxform = str
    config.add_section("Sequence")
    config.set("Sequence", "name", output_path.parts[-1])
    config.set("Sequence", "seqLength", str(len(ann["images"])))
    config.set("Sequence", "imWidth", str(ann["images"][0]["width"]))
    config.set("Sequence", "imHeight", str(ann["images"][0]["height"]))
    config.set("Sequence", "dynamic", str(is_moving(ann["images"])))
    with open(str(output_path / "seqinfo.ini"), "w") as info_output_file:
        config.write(info_output_file)


@click.command()
@click.option("--c", "--cores", "cores", default=4)
@click.option(
    "--ip",
    "--input_path",
    "input_path",
    default="/home/vy/university/thesis/datasets/MOTSynth/annotations",
    type=click.Path(exists=True),
    help="Path to raw annotations",
)
@click.option(
    "--op",
    "--output_path",
    "output_path",
    default="/home/vy/university/thesis/datasets/MOTSynth",
    type=click.Path(exists=True),
    help="Path to resulting gt annotations",
)
def main(input_path, output_path, cores):
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    input_path, output_path = Path(input_path), Path(output_path)
    output_path = output_path / "all"
    output_path.mkdir(parents=True, exist_ok=True)
    logging.log(logging.INFO, "Start generating keypoints gt")

    pool = Pool(cores)
    ann_names = sorted(input_path.glob("*"), key=lambda path: str(path))
    args = [(am, output_path) for am in ann_names]
    pool.map(multi_run_wrapper, args)
    logging.log(logging.INFO, "Saved to: {}".format(output_path))


if __name__ == "__main__":
    main()
