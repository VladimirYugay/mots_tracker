""" profiling MOTS dataset """
from mots_tracker import readers, utils, vis_utils
from mots_tracker.io_utils import get_instance, load_yaml


def profile_image(sample):
    vis_utils.plot_image(sample["image"])


def profile_panoptic(sample):
    vis_utils.plot_image(sample["panoptic_mask"])
    vis_utils.plot_image(sample["panoptic_image"])


def profile_depth(sample):
    vis_utils.plot_image(sample["depth"], image_type="depth")


def profile_scene_cloud(sample):
    print(sample["depth"].dtype, sample["image"].dtype, sample["intrinsics"].dtype)
    print(sample["depth"].shape, sample["image"].shape, sample["intrinsics"].shape)
    scene = utils.rgbd2ptcloud(sample["image"], sample["depth"], sample["intrinsics"])
    vis_utils.plot_ptcloud([scene])


def profile_instance_segmentation(sample):
    print(sample["instance_masks"].shape)
    # vis_utils.plot_image(sample["instance_masks"][0])
    vis_utils.plot_image_masks(sample["image"], sample["instance_masks"])


def profile_crps(sample):
    print(sample["current_correspondence"].shape)
    print(sample["next_correspondence"].shape)
    vis_utils.plot_image_2d_keypoints(
        sample["image"],
        sample["current_correspondence"][
            None,
        ],
    )
    # vis_utils.plot_image_2d_keypoints(
    #     sample["image"], sample["next_correspondence"][None, ])


def main():
    config_path = "./configs/debug_config.yaml"
    config = load_yaml(config_path)
    reader = get_instance(readers, "reader", config)
    seq_id, frame_id = "MOT16-14", 90
    sample = reader.read_sample(
        seq_id,
        frame_id,
    )
    print(sample.keys())
    # print(sample.keys())
    # profile_image(sample)
    # profile_panoptic(sample)
    # profile_depth(sample)
    # profile_instance_segmentation(sample)
    # profile_scene_cloud(sample)
    profile_crps(sample)


if __name__ == "__main__":
    main()
