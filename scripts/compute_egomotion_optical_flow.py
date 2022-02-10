""" Splits motsynth dataset into train, validation and test"""
import logging

import click
import numpy as np
import cv2

from mots_tracker import readers, utils, vis_utils
from mots_tracker.io_utils import get_instance, load_yaml

_logger = logging.getLogger(__name__)


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res

def get_displacement_map(flow):
    x = np.rint(flow[..., 0])
    y = np.rint(flow[..., 1])
    return x, y


def compute_egomotion(sample_left: dict, sample_right: dict) -> np.ndarray:
    """Compute egomotion between two consecutive frames

    Args:
        sample_left: dict with info about the frame
        sample_right: next dict with info about the frame
    Returns:
        np.ndarray: 4x4 transformation matrix
    """
    print(sample_left.keys())
    print(sample_left["optical_flow"].shape)
    print(sample_left["optical_flow"][..., 0].min(), sample_left["optical_flow"][..., 0].max())
    print(sample_left["optical_flow"][..., 1].min(), sample_left["optical_flow"][..., 1].max())

    x, y = get_displacement_map(sample_left["optical_flow"]git )
    cur_x, cur_y = np.indices((1080, 1920))
    next_x, next_y = cur_x + x, cur_y + y 

    import matplotlib.pyplot as plt 
    plt.imshow(sample_left["image"])
    plt.scatter(cur_x, cur_y, color='blue')
    plt.scatter(next_x, next_x, color='red')
    plt.savefig('lel.png')
    


def rigid_transform_3D(A, B, scale):
    assert A.shape == B.shape
    N = A.shape[0]  # total points
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))
    # dot is matrix multiplication for array
    if scale:
        H = np.transpose(BB) * AA / N
    else:
        # print(BB.T.shape, AA.shape)
        # H = np.matmul(BB.T, AA)
        H = np.transpose(BB) * AA
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T * U.T
    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T * U.T
    if scale:
        varA = np.var(A, axis=0).sum()
        c = 1 / (1 / varA * np.sum(S))  # scale factor
        t = -R * (centroid_B.T * c) + centroid_A.T
    else:
        c = 1
        t = -R * centroid_B.T + centroid_A.T
    R = R * c
    transformation = np.zeros((4, 4))
    transformation[3, 3] = 1
    transformation[:3, :3] = R
    transformation[:3, 3] = t.T
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
    "--op",
    "--output_path",
    "output_path",
    default="data/output",
    type=click.Path(exists=True),
    help="Output path of the for the split files",
)
def main(config_path, output_path):
    config = load_yaml(config_path)
    reader = get_instance(readers, "reader", config)
    for seq_id in config["reader"]["args"]["seq_ids"]:
        for frame_id in range(reader.sequence_info[seq_id]["length"] - 1):
            sample_left = reader.read_sample(seq_id, frame_id)
            sample_right = reader.read_sample(seq_id, frame_id)
            egomotion = compute_egomotion(sample_left, sample_right)
            print(egomotion)
            break
        break


if __name__ == "__main__":
    main()
