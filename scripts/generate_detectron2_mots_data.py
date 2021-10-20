""" Generates motsynth segmentation images  """
import json
import logging
import sys
from multiprocessing import Pool
from pathlib import Path

import click

from mots_tracker import readers
from mots_tracker.io_utils import get_instance, load_yaml
from mots_tracker.readers import reader_helpers

_logger = logging.getLogger(__name__)


def multi_run_wrapper(args):
    """ Unpacks argument for running on multiple cores """
    return make_coco(*args)


def make_coco(reader, seq_id, output_path):
    """Function to prepare jsons of MOTS data
    Args:
        reader (Reader): one of the readers implemented in readers module
        seq_id (str): id of the sequence
        output_path (Path): path to save output
    """
    coco_samples = []
    for frame_id in range(reader.sequence_info[seq_id]["length"]):
        print("Processing seq: {}, frame: {}".format(seq_id, frame_id))
        logging.log(
            logging.INFO, "Processing sequence: {}, frame: {}".format(seq_id, frame_id)
        )
        sample = reader.read_sample(seq_id, frame_id)
        annotations = []
        width = reader.sequence_info[seq_id]["img_width"]
        height = reader.sequence_info[seq_id]["img_height"]
        for raw_mask, bbox in zip(sample["raw_masks"], sample["boxes"]):
            annotations.append(
                {
                    "iscrowd": 0,
                    "segmentation": {"size": [height, width], "counts": raw_mask},
                    "bbox": [c for c in bbox],
                    "bbox_mode": 0,
                    "category_id": 0,
                }
            )
        img_name = reader.sequence_info[seq_id]["img_names"][frame_id]
        img_path = reader_helpers.id2imgpath(
            seq_id, img_name, reader.data_path / reader.mode
        )
        coco_sample = {
            "file_name": img_path,
            "image_id": seq_id + str(frame_id),
            "height": height,
            "width": width,
            "annotations": annotations,
        }
        coco_samples.append(coco_sample)
    with open(str(output_path) + "/{}_coco.json".format(seq_id), "w") as of:
        json.dump(coco_samples, of)


@click.command()
@click.option("--c", "--cores", "cores", default=4)
@click.option(
    "--op",
    "--output_path",
    "output_path",
    default="/home/vy/university/thesis/datasets/MOTSynth_COCO",
    type=click.Path(exists=True),
    help="Path to resulting segmentation masks",
)
@click.option(
    "--cp",
    "--config_path",
    "config_path",
    default="./configs/reproj_tracker_config.yaml",
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
    output_path = output_path

    config = load_yaml(config_path)
    reader = get_instance(readers, "reader", config)
    seq_ids = sorted(reader.sequence_info.keys())
    pool = Pool(cores)
    args = [(reader, seq_id, output_path) for seq_id in seq_ids]
    pool.map(multi_run_wrapper, args)


if __name__ == "__main__":
    main()
