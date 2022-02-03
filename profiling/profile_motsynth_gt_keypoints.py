""" profiling of gt keypoints from motsynth """
from mots_tracker import readers, utils, vis_utils
from mots_tracker.io_utils import get_instance, load_yaml


def main():
    """ visual profiling for generated motsynth keypoints """
    config_path = "./configs/keypoints_3d_tracker_config.yaml"
    config = load_yaml(config_path)
    reader = get_instance(readers, "reader", config)
    seq_id, frame_id = "000", 0
    sample = reader.read_sample(seq_id, frame_id)
    # vis_utils.plot_image_2d_keypoints(sample["image"], sample["keypoints_2d"])
    # vis_utils.plot_3d_keypoints(sample["keypoints_3d"
    print(sample["keypoints_3d"].shape)
    scene = utils.rgbd2ptcloud(sample["image"], sample["depth"], sample["intrinsics"])
    vis_utils.plot_3d_keypoints_o3d(
        sample["keypoints_3d"] + 1, range(sample["keypoints_3d"].shape[0]), scene
    )


if __name__ == "__main__":
    main()
