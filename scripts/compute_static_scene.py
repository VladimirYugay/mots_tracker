""" Splits motsynth dataset into train, validation and test"""
import copy
import logging
from pathlib import Path

import click
import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from tqdm import tqdm

from mots_tracker import readers, utils, vis_utils
from mots_tracker.io_utils import get_instance, load_yaml
from mots_tracker.readers.mot16_reader import DYNAMIC_SEQUENCES



@click.command()
@click.option(
    "--cp",
    "--config_path",
    "config_path",
    default="configs/debug_config.yaml",
    type=click.Path(exists=True),
    help="Path to the dataset: MOTS, MOTSynth, KITTI",
)
@click.option(
    "--op",
    "--output_path",
    "output_path",
    default="data/output",
    type=click.Path(exists=True),
    help="Output path of the for the split files",
)
def main(config_path, output_path):
    config = load_yaml(config_path)
    output_path = Path(output_path)
    reader = get_instance(readers, "reader", config)
    for seq_id in config["reader"]["args"]["seq_ids"]:

        if DYNAMIC_SEQUENCES[seq_id]:
            continue
        floor_alignment_path = "/usr/stud/yugay/MOT16/floor_alignment/"
        rotations = np.load("{}/{}_floor_rotations.npy".format(
            floor_alignment_path, seq_id))
        translations = np.load("{}{}_floor_translations.npy".format(
            floor_alignment_path, seq_id))        
        clouds = None
        for frame_id in tqdm(range(reader.sequence_info[seq_id]["length"] - 1)):
            if frame_id % 30 == 0:
                sample = reader.read_sample(seq_id, frame_id)
                panoptic = sample["panoptic_mask"]
                static_img = sample["image"]
                static_depth = sample["depth"]
                height, width = panoptic.shape

                mask = np.ones((height, width), dtype=np.int8)
                mask[panoptic == 0] = 0
                max_depth = 80
                mask[static_depth > max_depth] = 0
                grad_threshold = 30
                gradient = utils.compute_depth_gradient(sample["depth"])
                mask[gradient > grad_threshold] = 0

                static_img[mask == 0] = 0
                static_depth[mask == 0] = 0

                cloud = utils.rgbd2ptcloud(
                    static_img, static_depth, sample["intrinsics"]
                )
                # cloud.rotate(rotations[frame_id])
                # cloud.translate(translations[frame_id])

                if not clouds:
                    clouds = cloud
                else:
                    clouds += cloud

            # break
            # transformations.append(egomotion)
        o3d.io.write_point_cloud('scene_no_rot.pcd', clouds)
        # vis_utils.plot_ptcloud(clouds, show_frame=True)
        break
        transformations = np.array(transformations)
        np.save(filename, transformations)


if __name__ == "__main__":
    main()
