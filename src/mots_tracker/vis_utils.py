""" module for visualization utils """
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from matplotlib import colors as mcolors
from matplotlib import gridspec

from mots_tracker import utils
from mots_tracker.kalman_filters import BB2DKalmanFilter
from mots_tracker.kalman_filters.bb3d_kalman_filter import BB3DKalmanFilter
from mots_tracker.kalman_filters.median_kalman_filter import MedianKalmanFilter

np.random.seed(12)
# assume that there are no more than 2000 different object ids
# also try to make the colors different from each other
N_COLORS = 100
COLORS = np.array(
    [
        np.random.uniform(low=0.0, high=1, size=N_COLORS),
        np.random.uniform(low=0.2, high=1, size=N_COLORS),
        np.random.uniform(low=0.9, high=1, size=N_COLORS),
    ]
).T
M_COLORS = list(dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS).values())


def mcolor2rgb(mcolor):
    """converts matplotlib color to rgb [0, 255] color
    Args:
        mcolor (mcolor): matplotlib color
    Returns:
        ndarray: rgb [0, 255] color
    """
    return np.asarray(np.asarray(mcolors.to_rgb(mcolor)) * 255, dtype=np.uint8)


def colorize_patches(img_patches, color_weight=0.5, img_ids=None):
    """ colorizes pedestrian patches for visualization """
    colorized_patches = img_patches.copy()
    colorized_patches = colorized_patches.astype(np.float64)
    for patch_id in range(img_patches.shape[0]):
        color_id = patch_id if img_ids is None else img_ids[patch_id]
        color = np.asarray(
            np.asarray(mcolors.to_rgb(M_COLORS[color_id])) * 255, dtype=np.int
        )
        patch_mask = np.sum(colorized_patches[patch_id], axis=-1) != 0
        colorized_patches[patch_id, ...][patch_mask] *= 1 - color_weight
        colorized_patches[patch_id, ...][patch_mask] += color_weight * color
    return colorized_patches.astype(np.uint8)


def plot_topview(
    reader,
    seq_id,
    obj_ids,
    count=10,
    display_id=False,
    pixels_in_meter=40,
    depth_in_meter=0.14,
):
    """ plots top view trajectories of given pedestrians """
    plt.subplot(2, 1, 1)
    plt.title("Frame id: {}".format(1))
    sample = reader.read_sample(seq_id, 1)
    img = sample["image"].copy()
    for mask_id, mask in zip(sample["mask_ids"], sample["masks"]):
        if mask_id in obj_ids:
            img[mask == 1] = np.asarray(mcolors.to_rgb(M_COLORS[mask_id])) * 255
        if display_id and mask_id in obj_ids:
            plt.text(*utils.compute_mask_center(mask), mask_id)
    plt.imshow(img)

    plt.subplot(2, 1, 2)
    plt.xlabel("Horizontal axis (meters)")
    plt.ylabel("Depth (meters)")
    for i in range(count):
        sample = reader.read_sample(seq_id, i + 1)
        for mask_id, mask in zip(sample["mask_ids"], sample["masks"]):
            if mask_id not in obj_ids:
                continue
            depth = np.median(sample["depth"][mask == 1])
            _, x = np.nonzero(mask)
            c_x = x.min() + (x.max() - x.min()) // 2
            plt.scatter(
                c_x / pixels_in_meter, depth / depth_in_meter, color=M_COLORS[mask_id]
            )
    plt.show()


def plot_3d_trajectories(reader, seq_id, obj_ids=None, count=10):
    """ plots trajectory of projected object point clouds representations """
    points = defaultdict(list)
    for frame in range(count):
        sample = reader.read_sample(seq_id, frame)
        image, masks, depth = (
            sample["image"],
            sample["raw_masks"],
            sample["depth"],
        )
        img_patches, depth_patches = utils.patch_masks(image, masks), utils.patch_masks(
            depth, masks
        )
        clouds = [
            utils.rgbd2ptcloud(
                img_patch, depth_patch, reader.sequence_info[seq_id]["intrinsics"]
            )
            for img_patch, depth_patch in zip(img_patches, depth_patches)
        ]
        medians = np.array(
            [np.median(np.asarray(cld.points), axis=0) for cld in clouds]
        )
        for idx, median in zip(sample["mask_ids"], medians):
            if (obj_ids is not None and idx in obj_ids) or obj_ids is None:
                points[idx] += [median]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for obj_id in points.keys():
        obj_points = np.array(points[obj_id])
        ax.scatter(
            obj_points[:, 0],
            obj_points[:, 1],
            obj_points[:, 2],
            color=M_COLORS[obj_id],
            label="id: {}".format(obj_id),
        )
    plt.legend(loc="upper left")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()


def plot_image(image, image_type="RGB"):
    """ plots image with matplotlib """
    plt.figure()
    color_map = None
    if image_type != "RGB":
        color_map = plt.cm.get_cmap("plasma").reversed()
    plt.imshow(image, cmap=color_map)
    plt.show()  # display it
    return plt


def plot_image_boxes(image, boxes, boxes_ids=None):
    fig, ax = plt.subplots(1, dpi=96)
    ax.imshow(image)
    for i, box in enumerate(boxes):
        color = (
            M_COLORS[boxes_ids[i]]
            if boxes_ids is not None
            else np.random.uniform(0, 1, size=3)
        )
        rect = plt.Rectangle(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            fill=False,
            linewidth=1.0,
            color=color,
        )
        if boxes_ids is not None:
            plt.text(*utils.compute_box_center(box), boxes_ids[i], color=color)
        ax.add_patch(rect)
    plt.show()


def plot_image_masks(image, masks, mask_ids=None):
    img = image.copy()
    for i, mask in enumerate(masks):
        if mask_ids is not None:
            img[mask == 1] = np.asarray(mcolors.to_rgb(M_COLORS[mask_ids[i]])) * 255
            plt.text(*utils.compute_mask_center(mask), mask_ids[i])
        else:
            img[mask == 1] = 0
    plt.imshow(img)
    plt.show()


def plot_ptcloud(point_clouds):
    """visualizes point cloud
    Args:
        point_cloud (PointCloud): point cloud to visualize
    """
    # rotate down up
    if not isinstance(point_clouds, list):
        point_clouds = [point_clouds]
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1, origin=[0, 0, 0]
    )
    o3d.visualization.draw_geometries(point_clouds + [mesh_frame])


def vis_median_kalman_filter(reader, seq_id, obj_ids, steps=10):
    # ids for sequence 5 are 1, 125
    kalman_filters = {idx: MedianKalmanFilter(np.zeros(3), {}) for idx in obj_ids}
    estimates, gt, kalman_gains = (
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
    )
    steps = reader.sequence_info[seq_id]["length"] if steps == "all" else steps
    for step in range(steps):
        print("Processing frame {}".format(step + 1))
        # read the observations
        sample = reader.read_sample(seq_id, step)
        img_patches, depth_patches = utils.patch_masks(
            sample["image"], sample["masks"]
        ), utils.patch_masks(
            sample["image"],
            sample["depth"],
        )
        clouds = [
            utils.rgbd2ptcloud(
                img_patch, depth_patch, reader.sequence_info[seq_id]["intrinsics"]
            )
            for img_patch, depth_patch in zip(img_patches, depth_patches)
        ]
        medians = np.array(
            [np.median(np.asarray(cld.points), axis=0) for cld in clouds]
        )
        observations = {}
        for mask_id, median in zip(sample["mask_ids"], medians):
            if mask_id in obj_ids:
                observations[mask_id] = median

        # make update step, write the estimate, write the ground truth and kalman gain
        for idx, observation in observations.items():
            kalman_filters[idx].update(observation, {})
            kalman_gains[idx].append(np.sum(kalman_filters[idx].kf.K))
            estimates[idx].append(kalman_filters[idx].kf.x[:3][:, 0])
            gt[idx].append(observation)

    gs = gridspec.GridSpec(5, len(obj_ids))
    fig = plt.figure()
    for i, idx in enumerate(obj_ids):
        ax, ay, az, ae = (
            fig.add_subplot(gs[0, i]),
            fig.add_subplot(gs[1, i]),
            fig.add_subplot(gs[2, i]),
            fig.add_subplot(gs[3, i]),
        )
        akg = fig.add_subplot(gs[4, i])
        ax.title.set_text("Object id: {}".format(idx))
        truth, est, kg = (
            np.array(gt[idx]),
            np.array(estimates[idx]),
            np.array(kalman_gains[idx]),
        )
        error = np.linalg.norm(truth - est, axis=1)
        x = np.arange(truth.shape[0])

        ax.plot(x, truth[:, 0], "g", x, est[:, 0], "bs-")
        ay.plot(x, truth[:, 1], "g", x, est[:, 1], "bs-")
        lgt, lest = az.plot(x, truth[:, 2], "g", x, est[:, 2], "bs-")
        le = ae.plot(x, error, "red", label="error")
        lkg = akg.plot(x, kg, "orange", label="kalman gain")
        if i == 0:
            ax.set_ylabel("X")
            ay.set_ylabel("Y")
            az.set_ylabel("Z")
            ae.set_ylabel("Error")
            akg.set_ylabel("Kalman Gain")
        az.set_xlabel("Step")
        print(
            "Sequence: {}, object id: {}, MSE: {}".format(seq_id, idx, np.mean(error))
        )
    fig.legend(
        (lgt, lest, le[0], lkg[0]),
        ("Ground truth", "Estimate", "Error", "Kalman Gain"),
        "upper left",
    )
    plt.show()


def vis_bbox3d_kalman_filter(reader, seq_id, obj_ids, steps=10):
    kalman_filters = {idx: BB3DKalmanFilter(np.zeros(6), {}) for idx in obj_ids}
    estimates, gt, kalman_gains = (
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
    )
    steps = reader.sequence_info[seq_id]["length"] if steps == "all" else steps
    for step in range(steps):
        print("Processing frame {}".format(step + 1))
        # read the observations
        sample = reader.read_sample(seq_id, step)
        img_patches, depth_patches = (
            utils.patch_masks(sample["image"], sample["masks"]),
            utils.patch_masks(sample["depth"], sample["masks"]),
        )
        clouds = [
            utils.rgbd2ptcloud(
                img_patch, depth_patch, reader.sequence_info[seq_id]["intrinsics"]
            )
            for img_patch, depth_patch in zip(img_patches, depth_patches)
        ]
        observations = {}
        for cloud, mask_id in zip(clouds, sample["mask_ids"]):
            try:
                box = cloud.get_axis_aligned_bounding_box()
                bb3d = np.concatenate((box.get_center(), box.get_extent()))
                if mask_id in obj_ids:
                    observations[mask_id] = bb3d
            except Exception:
                continue

        # make update step, write the estimate, write the ground truth and kalman gain
        for idx, observation in observations.items():
            kalman_filters[idx].update(observation, {})
            kalman_gains[idx].append(np.sum(kalman_filters[idx].kf.K))
            estimates[idx].append(kalman_filters[idx].kf.x[:6][:, 0])
            gt[idx].append(observation)

    gs = gridspec.GridSpec(8, len(obj_ids))
    fig = plt.figure()
    for i, idx in enumerate(obj_ids):
        ax, ay, az = (
            fig.add_subplot(gs[0, i]),
            fig.add_subplot(gs[1, i]),
            fig.add_subplot(gs[2, i]),
        )
        aw, ah, al = (
            fig.add_subplot(gs[3, i]),
            fig.add_subplot(gs[4, i]),
            fig.add_subplot(gs[5, i]),
        )
        ae, akg = fig.add_subplot(gs[6, i]), fig.add_subplot(gs[7, i])
        ax.title.set_text("Object id: {}".format(idx))
        truth, est, kg = (
            np.array(gt[idx]),
            np.array(estimates[idx]),
            np.array(kalman_gains[idx]),
        )
        error = np.linalg.norm(truth - est, axis=1)
        x = np.arange(truth.shape[0])

        ax.plot(x, truth[:, 0], "g", x, est[:, 0], "bs-")
        ay.plot(x, truth[:, 1], "g", x, est[:, 1], "bs-")
        lgt, lest = az.plot(x, truth[:, 2], "g", x, est[:, 2], "bs-")
        aw.plot(x, truth[:, 3], "g", x, est[:, 3], "bs-")
        ah.plot(x, truth[:, 4], "g", x, est[:, 4], "bs-")
        al.plot(x, truth[:, 5], "g", x, est[:, 5], "bs-")
        le = ae.plot(x, error, "red", label="error")
        lkg = akg.plot(x, kg, "orange", label="kalman gain")
        if i == 0:
            ax.set_ylabel("X")
            ay.set_ylabel("Y")
            az.set_ylabel("Z")
            aw.set_ylabel("Width")
            ah.set_ylabel("Height")
            al.set_ylabel("Length")
            ae.set_ylabel("Error")
            akg.set_ylabel("Kalman Gain")
        az.set_xlabel("Step")
        print(
            "Sequence: {}, object id: {}, MSE: {}".format(seq_id, idx, np.mean(error))
        )
    fig.legend(
        (lgt, lest, le[0], lkg[0]),
        ("Ground truth", "Estimate", "Error", "Kalman Gain"),
        "upper left",
    )
    plt.show()


def vis_bbox2d_kalman_filter(reader, seq_id, obj_ids, steps=10):
    # ids for sequence 5 are 1, 125
    kalman_filters = {
        idx: BB2DKalmanFilter(np.array([417.00, 458.00, 39.00, 83.00]), {})
        for idx in obj_ids
    }
    estimates, gt, kalman_gains = (
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
    )
    steps = reader.sequence_info[seq_id]["length"] if steps == "all" else steps
    for step in range(steps):
        print("Processing frame {}".format(step + 1))
        # read the observations
        sample = reader.read_sample(seq_id, step)
        observations = {}
        for box_id, box in zip(sample["box_ids"], sample["boxes"]):
            if box_id in obj_ids:
                observations[box_id] = box

        # make update step, write the estimate, write the ground truth and kalman gain
        for idx, observation in observations.items():
            kalman_filters[idx].update(observation, {})
            kalman_gains[idx].append(np.sum(kalman_filters[idx].kf.K))
            estimates[idx].append(kalman_filters[idx].get_state()[0])
            gt[idx].append(observation)
    gs = gridspec.GridSpec(6, len(obj_ids))
    fig = plt.figure()
    for i, idx in enumerate(obj_ids):
        ax, ay, aa, ar = (
            fig.add_subplot(gs[0, i]),
            fig.add_subplot(gs[1, i]),
            fig.add_subplot(gs[2, i]),
            fig.add_subplot(gs[3, i]),
        )
        ae, akg = fig.add_subplot(gs[4, i]), fig.add_subplot(gs[5, i])
        ax.title.set_text("Object id: {}".format(idx))
        truth, est, kg = (
            np.array(gt[idx]),
            np.array(estimates[idx]),
            np.array(kalman_gains[idx]),
        )
        error = np.linalg.norm(truth - est, axis=1)
        x = np.arange(truth.shape[0])

        ax.plot(x, truth[:, 0], "g", x, est[:, 0], "bs-")
        ay.plot(x, truth[:, 1], "g", x, est[:, 1], "bs-")
        aa.plot(x, truth[:, 2], "g", x, est[:, 2], "bs-")
        lgt, lest = ar.plot(x, truth[:, 3], "g", x, est[:, 3], "bs-")
        le = ae.plot(x, error, "red", label="error")
        lkg = akg.plot(x, kg, "orange", label="kalman gain")
        if i == 0:
            ax.set_ylabel("X")
            ay.set_ylabel("Y")
            aa.set_ylabel("Area")
            ar.set_ylabel("Ratio")
            ae.set_ylabel("Error")
            akg.set_ylabel("Kalman Gain")
        ar.set_xlabel("Step")
        print(
            "Sequence: {}, object id: {}, MSE: {}".format(seq_id, idx, np.mean(error))
        )
    fig.legend(
        (lgt, lest, le[0], lkg[0]),
        ("Ground truth", "Estimate", "Error", "Kalman Gain"),
        "upper left",
    )
    plt.show()
