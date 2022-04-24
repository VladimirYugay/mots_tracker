""" profiling gt bb generation from motsynth """
from functools import partial

import cv2
import numpy as np
from tqdm import tqdm

from mots_tracker import readers, utils, vis_utils
from mots_tracker.io_utils import get_instance, load_yaml
from mots_tracker.trackers import tracker_helpers


def profile_clouds(reader, seq_id, frame_id):
    """ See the clouds """
    sample = reader.read_sample(seq_id, frame_id)
    cloud_filter = partial(tracker_helpers.depth_median_filter, radius=0.3)
    clouds = utils.compute_mask_clouds(sample, cloud_filter)
    vis_utils.plot_ptcloud(clouds, False)


def profile_depth(reader, seq_id, frame_id):
    """ See the depth """
    sample = reader.read_sample(seq_id, frame_id)
    vis_utils.plot_image(sample["depth"], image_type="Depth")


def profile_scene_cloud(reader, seq_id, frame_id):
    """ See the clouds """
    sample = reader.read_sample(seq_id, frame_id)
    # depth = load_motsynth_depth_image(
    #       '/home/vy/university/thesis/datasets/000/gt_depth/0000.png')
    depth = sample["depth"]
    scene = utils.rgbd2ptcloud(sample["image"], depth, sample["intrinsics"])
    vis_utils.plot_ptcloud(scene, False)


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


def profile_new_depth_boxes(sample):
    """ Profile new box computation for the raw depth maps """
    from mots_tracker.trackers import tracker_helpers

    cloud_filter = partial(tracker_helpers.depth_median_filter, radius=1)
    clouds = utils.compute_mask_clouds(sample, cloud_filter)
    boxes = utils.compute_axis_aligned_bbs(clouds)
    vis_utils.plot_ptcloud(clouds + list(boxes.values()))
    print(sample["masks"].shape, len(clouds))


def profile_new_depth_rotated_boxes(reader, seq_id, frame_id):
    """ Profile how new boxes will be rotated with the new depth """
    np.set_printoptions(suppress=True, precision=3)
    sample_left = reader.read_sample(seq_id, frame_id)
    sample_right = reader.read_sample(seq_id, frame_id + 1)
    cloud_filter = partial(tracker_helpers.depth_median_filter, radius=1)

    T_right = sample_right["egomotion"]
    clouds_left = utils.compute_mask_clouds(sample_left, cloud_filter)

    clouds_right = utils.compute_mask_clouds(sample_right, cloud_filter)
    boxes_right = list(utils.compute_axis_aligned_bbs(clouds_right).values())

    vis_utils.plot_ptcloud(clouds_left + boxes_right, False)

    clouds_right = utils.compute_mask_clouds(sample_right, cloud_filter)
    for cloud in clouds_right:
        cloud.transform(T_right)
    boxes_right = list(utils.compute_axis_aligned_bbs(clouds_right).values())
    vis_utils.plot_ptcloud(clouds_left + boxes_right, False)


def main():
    """ visual profiling for generated motsynth bb """
    config_path = "./configs/median_tracker_config.yaml"
    config = load_yaml(config_path)
    reader = get_instance(readers, "reader", config)
    seq_id, frame_id = "002", 0
    sample = reader.read_sample(seq_id, frame_id)
    # profile_new_depth_boxes(sample)
    ids = sample["mask_ids"]
    target_id = 399453
    print(ids[8])
    mask = sample["masks"][ids == 399453]
    # vis_utils.plot_image_masks(
    #     sample["image"], sample["masks"], np.arange(ids.shape[0]))
    # vis_utils.plot_image_masks(
    #     sample["image"], mask)    

    stats = []
    for frame_id in tqdm(range(0, 240)):
        sample = reader.read_sample(seq_id, frame_id)
        mask = sample["masks"][sample["mask_ids"] == target_id][0]
        depth_patch = sample["depth"]
        depth_patch[mask == 0] = 0
        q5 = np.percentile(depth_patch[depth_patch != 0], 5)
        q95 = np.percentile(depth_patch[depth_patch != 0], 95)
        depth_patch[depth_patch < q5] = 0
        depth_patch[depth_patch > q95] = 0 
        
        img = sample["image"]
        img[depth_patch == 0] = 0

        cloud = utils.rgbd2ptcloud(img, depth_patch, sample["intrinsics"])
        cloud = np.asarray(cloud.points)

        stats.append([
            np.mean(cloud), 
            np.median(cloud), 
            np.std(cloud), 
            np.percentile(cloud, 25),
            np.percentile(cloud, 75)])

    import matplotlib.pyplot as plt 
    stats = np.array(stats)
    x = np.arange(stats.shape[0])
    names = ["mean", "median", "std", "q25", "q75"]
    for i, name in enumerate(names):
        plt.plot(x, stats[:, i], label=name)
    plt.legend()
    plt.savefig('stats')

    # vis_utils.plot_image_boxes(sample["image"], sample["boxes"][[0, 1]], np.arange(2))

    # vis_utils.plot_image_masks(sample['image'], sample['masks'])
    # profile_new_depth_boxes(sample)
    # profile_scene_cloud(reader, seq_id, frame_id)
    # profile_clouds(reader, seq_id, frame_id)
    # profile_new_depth_rotated_boxes(reader, seq_id, frame_id)


if __name__ == "__main__":
    main()
