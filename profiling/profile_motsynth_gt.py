""" profiling gt bb generation from motsynth """
import cv2
import numpy as np

from mots_tracker import utils, vis_utils
from mots_tracker.readers import MOTSynthReader
from mots_tracker.trackers import tracker_helpers


def profile_depth(reader, seq_id, frame_id):
    """ See the depth """
    sample = reader.read_sample(seq_id, frame_id)
    vis_utils.plot_image(sample["depth"], image_type="Depth")


def profile_clouds(reader, seq_id, frame_id):
    """ See the clouds """
    sample = reader.read_sample(seq_id, frame_id)
    clouds = utils.compute_mask_clouds(sample, tracker_helpers.depth_median_filter)
    vis_utils.plot_ptcloud(clouds, False)


def profile_scene_cloud(reader, seq_id, frame_id):
    """ See the clouds """
    sample = reader.read_sample(seq_id, frame_id)
    # depth = load_motsynth_depth_image(
    #       '/home/vy/university/thesis/datasets/000/gt_depth/0000.png')
    depth = sample["depth"]
    scene = utils.rgbd2ptcloud(sample["image"], depth, sample["intrinsics"])
    vis_utils.plot_ptcloud(scene)


def convert_depth(img_path):
    abs_min = 1008334389
    abs_max = 1067424357
    x = np.fromfile(img_path, dtype="uint32")
    x = x[4:]
    x = np.clip(x, abs_min, abs_max)
    x = (x - abs_min) / (abs_max - abs_min)
    x = np.uint8(x * 255)
    x = x.reshape(1080, 1920)
    return x


def read_depthmap(name):
    n = 1.04187
    f = 800

    abs_min = 1008334389
    abs_max = 1067424357

    depth = cv2.imread(name)[:, :, 0]
    depth = np.uint32(depth)
    depth = depth / 255
    depth = (depth * (abs_max - abs_min)) + abs_min
    depth = depth.astype("uint32")
    depth.dtype = "float32"

    y = (-(n * f) / (n - f)) / (depth - (n / (n - f)))
    y = y.reshape((1080, 1920))
    return y


def main():
    """ visual profiling for generated motsynth bb """
    config = {
        "depth_path": "gt_depth",
        "egomotion_path": "egomotion",
        "read_masks": True,
        "read_boxes": True,
        "gt_path": "/home/vy/university/thesis/datasets/MOTSynth_annotations/all",
        "split_path": None,
    }
    root_path = "/home/vy/university/thesis/datasets/MOTSynth"
    reader = MOTSynthReader(root_path, config)
    seq_id, frame_id = "000", 0

    profile_scene_cloud(reader, seq_id, frame_id)


if __name__ == "__main__":
    main()
