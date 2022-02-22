""" Splits motsynth dataset into train, validation and test"""
from pathlib import Path

import click
import numpy as np
import pandas as pd
from tqdm import tqdm

from mots_tracker import readers, utils
from mots_tracker.io_utils import get_instance, load_yaml
from mots_tracker.trackers.tracker_helpers import depth_median_filter
from mots_tracker.readers.mot16_reader import DYNAMIC_SEQUENCES


def compute_clouds(sample: dict, rotation: np.ndarray, translation: np.ndarray) -> dict:
    """Computes 3d positions of the pedestrian point clouds
    Args:
        sample: sample dictionary
    Returns:
        dictionary with the 3d pose of the flooar aligned point cloud
    """

    depth = sample["depth"].copy()
    # depth[utils.compute_depth_gradient(depth) > 30] = 0
    img_patches = utils.patch_masks(sample["image"], sample["instance_masks"])
    depth_patches = utils.patch_masks(depth, sample["instance_masks"])
    clouds = [
        utils.rgbd2ptcloud(
            img_patch, depth_patch, sample["intrinsics"], depth_median_filter
        )
        for img_patch, depth_patch in zip(img_patches, depth_patches)
    ]
    clouds = [c for c in clouds if c.has_points()]
    for cloud in clouds:
        cloud.rotate(rotation)
        cloud.translate(translation)
    return clouds


def compute_cloud_params(clouds: list) -> tuple:
    """Computes parameters of the point clouds
    Args:
        clouds: list of point clouds
    Returns:
        parameters needed
    """
    medians = np.array([np.median(np.asarray(cld.points), axis=0) for cld in clouds])
    heights = [
        abs(np.asarray(cloud.points)[:, 1].max() - np.asarray(cloud.points)[:, 1].min())
        for cloud in clouds
    ]
    return medians, heights


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
    "--fap",
    "--floor_alignment_path",
    "floor_alignment_path",
    default=None,
    type=click.Path(exists=True),
    help="Path for the floor transformations",
)
@click.option(
    "--op",
    "--output_path",
    "output_path",
    default="data/output",
    type=click.Path(exists=True),
    help="Output path of the for the split files",
)
def main(config_path, output_path, floor_alignment_path):
    config = load_yaml(config_path)
    output_path = Path(output_path)
    reader = get_instance(readers, "reader", config)
    filename = str(output_path / "trajectories.csv")

    for seq_id in config["reader"]["args"]["seq_ids"][1:]:
        
        if DYNAMIC_SEQUENCES[seq_id]:
            continue 

        filename = str(output_path / "{}_trajectories.csv".format(seq_id))
        seq_trajectory = []
        rotations, translations = None, None
        if floor_alignment_path is not None:
            rotations = np.load("{}/{}_floor_rotations.npy".format(
                floor_alignment_path, seq_id))
            translations = np.load("{}{}_floor_translations.npy".format(
                floor_alignment_path, seq_id))

        print("Processing", seq_id)
        for frame_id in tqdm(range(reader.sequence_info[seq_id]["length"])):
            sample = reader.read_sample(seq_id, frame_id)
            if rotations is not None and translations is not None:
                clouds = compute_clouds(
                    sample, rotations[frame_id], translations[frame_id]
                )
            else:
                clouds = compute_clouds(sample, np.eye(3), np.zeros(3))
            medians, heights = compute_cloud_params(clouds)
            for i, (m, h) in enumerate(zip(medians, heights)):
                seq_trajectory.append(
                    {   
                        "mask_id": i + 1,
                        "seq_id": seq_id,
                        "frame_id": frame_id,
                        "median_x": m[0],
                        "median_y": m[1],
                        "median_z": m[2],
                        "height": h,
                    }
                )
        df = pd.DataFrame(seq_trajectory)
        df.to_csv(filename)


if __name__ == "__main__":
    main()
