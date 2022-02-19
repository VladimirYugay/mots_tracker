""" Splits motsynth dataset into train, validation and test"""
import logging
from pathlib import Path

import click
import numpy as np

# import open3d as o3d
from tqdm import tqdm

from mots_tracker import readers, utils, vis_utils
from mots_tracker.io_utils import get_instance, load_yaml
from mots_tracker.readers.mot16_reader import DYNAMIC_SEQUENCES

_logger = logging.getLogger(__name__)


def compute_egomotion(source_sample: dict, target_sample: dict) -> np.ndarray:
    """Compute egomotion between two consecutive frames

    Args:
        source_sample: dict with info about the frame
        target_sample: next dict with info about the frame
    Returns:
        np.ndarray: 4x4 transformation matrix
    """

    # verify_ptcd_order(source_sample)

    height, width, _ = source_sample["image"].shape
    flow = source_sample["optical_flow"]
    hflow, vflow = flow[..., 0], flow[..., 1]

    spanoptic, tpanoptic = (
        source_sample["panoptic_mask"],
        target_sample["panoptic_mask"],
    )

    # filter invalid classes and unknown flow values
    mask = np.ones((height, width), dtype=np.int8)
    mask[np.logical_or(spanoptic == 0, tpanoptic == 0)] = 0
    mask[np.logical_or(hflow == 1e9, vflow == 1e9)] = 0

    # filter too far depth regions
    max_depth = 80
    mask[
        np.logical_or(
            source_sample["depth"] > max_depth, target_sample["depth"] > max_depth
        )
    ] = 0

    # filter based on stable gradient
    grad_threshold = 30  # since lower kills too much rigid objects around
    sgradient = utils.compute_depth_gradient(source_sample["depth"])
    tgradient = utils.compute_depth_gradient(target_sample["depth"])
    mask[np.logical_or(sgradient > grad_threshold, tgradient > grad_threshold)] = 0

    # get the ids of interest
    srows, scols = np.where(mask != 0)
    hdisp, vdisp = hflow[srows, scols], vflow[srows, scols]
    hdisp, vdisp = np.rint(hdisp).astype(np.int32), np.rint(vdisp).astype(np.int32)
    trows, tcols = srows + vdisp, scols + hdisp

    # remove the out of boundaries pixels
    valid_pixels = (tcols < width) & (trows < height)
    srows, scols = srows[valid_pixels], scols[valid_pixels]
    trows, tcols = trows[valid_pixels], tcols[valid_pixels]

    # we want to preserve the semantic class between the pixels
    valid_cls = spanoptic[srows, scols] == tpanoptic[trows, tcols]
    srows, scols = srows[valid_cls], scols[valid_cls]
    trows, tcols = trows[valid_cls], tcols[valid_cls]

    source_cloud = utils.rgbd2ptcloud(
        source_sample["image"], source_sample["depth"], source_sample["intrinsics"]
    )
    target_cloud = utils.rgbd2ptcloud(
        target_sample["image"], target_sample["depth"], target_sample["intrinsics"]
    )

    # align both point clouds to the ground
    # R_source, t_source = source_sample["R_floor"], source_sample["t_floor"]
    # source_cloud.rotate(R_source)
    # source_cloud.translate(t_source)
    # source_shift = np.asarray(source_cloud.points)[:, 1].min()
    # source_cloud.translate([0, -source_shift, 0])

    # R_target, t_target = target_sample["R_floor"], target_sample["t_floor"]
    # target_cloud.rotate(R_target)
    # target_cloud.translate(t_target)
    # target_shift = np.asarray(target_cloud.points)[:, 1].min()
    # target_cloud.translate(np.array([0, -target_shift, 0]))

    # select correspondences
    source_cloud_pts = np.asarray(source_cloud.points)
    target_cloud_pts = np.asarray(target_cloud.points)

    source_ids = srows * width + scols
    target_ids = trows * width + tcols
    source_pts = source_cloud_pts[source_ids]
    target_pts = target_cloud_pts[target_ids]

    # compute the transformation along Y axis
    source_pts = np.asmatrix(source_pts[:, [0, 2]])
    target_pts = np.asmatrix(target_pts[:, [0, 2]])
    T = utils.rigid_transform_3D(source_pts, target_pts, False)

    # it's 3x3 since rotation was estimated only along Y axis
    transformation = np.eye(4)
    transformation[:3, :3] = np.array(
        [[T[0][0], 0, T[0][1]], [0, 1, 0], [T[1][0], 0, T[1][1]]]
    )
    transformation[0][3] = T[0][2]
    transformation[2][3] = T[1][2]

    return transformation


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
    "--floop_alignment_path",
    "floop_alignment_path",
    type=click.Path(exists=True),
    help="Output path of the for the split files",
)
@click.option(
    "--op",
    "--output_path",
    "output_path",
    default="data/output",
    type=click.Path(exists=True),
    help="Output path of the for the split files",
)
def main(config_path, floop_alignment_path, output_path):
    config = load_yaml(config_path)
    output_path = Path(output_path)
    floop_alignment_path = Path(floop_alignment_path)

    reader = get_instance(readers, "reader", config)
    for seq_id in config["reader"]["args"]["seq_ids"]:

        if not DYNAMIC_SEQUENCES[seq_id]:
            continue

        rotations = np.load(
            str(floop_alignment_path / "{}_floor_rotations.npy".format(seq_id))
        )
        translations = np.load(
            str(floop_alignment_path / "{}_floor_translations.npy".format(seq_id))
        )

        print("Processing", seq_id)
        filename = "{}_egomotion.npy".format(seq_id)
        filename = str(output_path / filename)
        transformations = []
        for frame_id in tqdm(range(0, reader.sequence_info[seq_id]["length"] - 1)):

            source_sample = reader.read_sample(seq_id, frame_id)
            target_sample = reader.read_sample(seq_id, frame_id + 30)

            source_sample["R_floor"] = rotations[frame_id]
            source_sample["t_floor"] = translations[frame_id]
            target_sample["R_floor"] = rotations[frame_id + 1]
            target_sample["t_floor"] = translations[frame_id + 1]

            egomotion = compute_egomotion(source_sample, target_sample)
            transformations.append(egomotion)

            sc = utils.rgbd2ptcloud(
                source_sample["image"],
                source_sample["depth"],
                source_sample["intrinsics"],
            )
            tc = utils.rgbd2ptcloud(
                target_sample["image"],
                target_sample["depth"],
                target_sample["intrinsics"],
            )
            # tc.paint_uniform_color([0, 1, 0])

            vis_utils.plot_ptcloud([sc, tc], True)

            tc.transform(egomotion)

            vis_utils.plot_ptcloud([sc, tc], True)

            np.set_printoptions(suppress=True, precision=3)
            print(egomotion)
            break

        break

        transformations = np.array(transformations)
        np.save(filename, transformations)


if __name__ == "__main__":
    main()
