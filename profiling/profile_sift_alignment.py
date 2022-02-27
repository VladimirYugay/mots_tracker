""" profiling MOTS dataset """
import cv2

# import matplotlib.pyplot as plt
import numpy as np

from mots_tracker import readers, utils, vis_utils
from mots_tracker.io_utils import get_instance, load_yaml

np.set_printoptions(suppress=True, precision=3)
np.random.seed(42)


def preprocess_samples(source_sample, target_sample):
    height, width, _ = source_sample["image"].shape

    spanoptic, tpanoptic = (
        source_sample["panoptic_mask"],
        target_sample["panoptic_mask"],
    )

    # filter invalid classes and unknown flow values
    mask = np.ones((height, width), dtype=np.int8)
    mask[np.logical_or(spanoptic == 0, tpanoptic == 0)] = 0

    # filter too far depth regions
    max_depth = 100
    mask[
        np.logical_or(
            source_sample["depth"] > max_depth, target_sample["depth"] > max_depth
        )
    ] = 0

    # filter based on stable gradient
    # grad_threshold = 50  # since lower kills too much rigid objects around
    # sgradient = utils.compute_depth_gradient(source_sample["depth"])
    # tgradient = utils.compute_depth_gradient(target_sample["depth"])
    # mask[np.logical_or(sgradient > grad_threshold, tgradient > grad_threshold)] = 0
    return mask


def get_features(a, b):

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(a, None)
    kp2, des2 = sift.detectAndCompute(b, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)
    matched_pixels = []
    for match in matches:
        img1_idx = match[0].queryIdx
        img2_idx = match[0].trainIdx
        (col1, row1) = kp1[img1_idx].pt
        (col2, row2) = kp2[img2_idx].pt
        matched_pixels.append([row1, col1, row2, col2])

    matchesMask = [[0, 0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matchesMask[i] = [1, 0]

    matched_pixels = []
    for a, b in matches:
        if a.distance >= 0.7 * b.distance:
            continue
        img1_idx = a.queryIdx
        img2_idx = a.trainIdx
        (col1, row1) = kp1[img1_idx].pt
        (col2, row2) = kp2[img2_idx].pt
        matched_pixels.append([row1, col1, row2, col2])

    # draw_params = dict(
    #     matchColor = (0,255,0), singlePointColor = (255,0,0),
    #     matchesMask = matchesMask, flags = cv2.DrawMatchesFlags_DEFAULT)
    # c = cv2.drawMatchesKnn(a, kp1, b, kp2, matches, None, **draw_params)
    # plt.imshow(c)
    # plt.show()
    return np.array(matched_pixels)


def profile_sift(source_sample, target_sample):

    height, width, _ = source_sample["image"].shape
    mask = preprocess_samples(source_sample, target_sample)

    simg, timg = source_sample["image"], target_sample["image"]
    # sdepth, tdepth = source_sample["depth"], target_sample["depth"]

    # simg[mask == 0] = 0
    # timg[mask == 0] = 0
    # tdepth[mask == 0] = 0
    # sdepth[mask == 0] = 0

    matched_pixels = get_features(simg, timg).astype(np.int32)

    srows, scols = matched_pixels[:, 0], matched_pixels[:, 1]
    trows, tcols = matched_pixels[:, 2], matched_pixels[:, 3]
    print(srows.dtype, scols.dtype)
    # remove pixels belonging to classes that we ignore
    valid_pixels = np.logical_and(mask[srows, scols] == 1, mask[trows, tcols] == 1)
    srows, scols = srows[valid_pixels], scols[valid_pixels]
    trows, tcols = trows[valid_pixels], tcols[valid_pixels]

    # ids = np.random.randint(0, srows.shape[0], 100)
    # fig, (sax, tax) = plt.subplots(nrows=1, ncols=2)
    # sax.imshow(simg)
    # sax.scatter(scols[ids], srows[ids], c=ids)
    # tax.imshow(timg)
    # tax.scatter(tcols[ids], trows[ids], c=ids)
    # plt.show()

    source_cloud = utils.rgbd2ptcloud(
        source_sample["image"], source_sample["depth"], source_sample["intrinsics"]
    )
    target_cloud = utils.rgbd2ptcloud(
        target_sample["image"], target_sample["depth"], target_sample["intrinsics"]
    )

    source_ids = srows * width + scols
    target_ids = trows * width + tcols

    # visualize 3d points to align
    a = source_cloud.select_down_sample(source_ids)
    a.paint_uniform_color([0, 1, 0])
    b = target_cloud.select_down_sample(target_ids)
    b.paint_uniform_color([0, 0, 1])
    vis_utils.plot_ptcloud([a, b])

    source_cloud_pts = np.asarray(source_cloud.points)
    target_cloud_pts = np.asarray(target_cloud.points)
    source_cloud_pts = source_cloud_pts[source_ids]
    target_cloud_pts = target_cloud_pts[target_ids]

    T = utils.rigid_transform_3D(
        np.asmatrix(source_cloud_pts[:, [0, 2]]),
        np.asmatrix(target_cloud_pts[:, [0, 2]]),
        False,
    )
    transformation = np.eye(4)
    transformation[:3, :3] = np.array(
        [[T[0][0], 0, T[0][1]], [0, 1, 0], [T[1][0], 0, T[1][1]]]
    )
    transformation[0][3] = T[0][2]
    transformation[2][3] = T[1][2]

    # visualize points after align
    a = source_cloud.select_down_sample(source_ids)
    b = target_cloud.select_down_sample(target_ids)
    a.paint_uniform_color([0, 1, 0])
    b.paint_uniform_color([0, 0, 1])
    b.transform(transformation)
    vis_utils.plot_ptcloud([a, b])


def main():
    config_path = "./configs/debug_config.yaml"
    config = load_yaml(config_path)
    reader = get_instance(readers, "reader", config)
    seq_id, frame_id = "MOT16-14", 0  # 300 for challenge
    source_sample = reader.read_sample(
        seq_id,
        frame_id,
    )
    target_sample = reader.read_sample(
        seq_id,
        frame_id + 10,
    )

    # floor_align_path = "/home/vy/university/thesis/datasets/MOT16/floor_alignment/"
    # floor_rots = np.load(floor_align_path + "MOT16-14_floor_rotations.npy")
    # floor_trans = np.load(floor_align_path + "MOT16-14_floor_translations.npy")

    profile_sift(source_sample, target_sample)


if __name__ == "__main__":
    main()
