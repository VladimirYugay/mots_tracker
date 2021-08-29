""" module for visualization utils """
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from matplotlib import colors as mcolors
from matplotlib import patches
from open3d.open3d.geometry import PointCloud
from PIL import Image

from mots_tracker import utils

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


def plot_ptcloud(point_clouds, show_frame=True):
    """visualizes point cloud
    Args:
        point_cloud (PointCloud): point cloud to visualize
    """
    # rotate down up
    if not isinstance(point_clouds, list):
        point_clouds = [point_clouds]
    if show_frame:
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=1, origin=[0, 0, 0]
        )
        point_clouds = point_clouds + [mesh_frame]
    o3d.visualization.draw_geometries(point_clouds)


def plot_box_patch(axis, box, box_id=None):
    """Adds box patch to an axis
    Args:
        axis (Axis): matplotlib axis
        box (ndarray): bounding box in (x1, y1, x2, y2) format
        box_id (int): id of the bounding box
    """
    if box_id is None:
        box_id = 0
    color = M_COLORS[int(box_id) % len(M_COLORS)]
    axis.add_patch(
        patches.Rectangle(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            fill=False,
            lw=3,
            color=color,
        )
    )
    axis.text(*utils.compute_box_center(box), box_id, color=color)


def play_clouds_movement(list_of_pcds: []):
    """Plays movement of pedestrain point clouds
    Args:
        list_of_pcds list(list): lists of points clouds of shape n_frames x n_clouds
    """

    def copy_pcds(pcds):
        new_pcds = [PointCloud() for _ in pcds]
        for new_pcd, pcd in zip(new_pcds, pcds):
            new_pcd.colors = pcd.colors
            new_pcd.points = pcd.points
        return new_pcds

    def reset_motion(vis):
        play_clouds_movement.index = 0
        for fpcd in play_clouds_movement.frame_pcds:
            vis.remove_geometry(fpcd)
        play_clouds_movement.frame_pcds = copy_pcds(list_of_pcds[0])
        for fpcd in play_clouds_movement.frame_pcds:
            vis.add_geometry(fpcd)
        vis.get_view_control().rotate(0, 500)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(1)
        vis.register_animation_callback(forward)
        return False

    def forward(vis):
        pm = play_clouds_movement
        if pm.index < len(list_of_pcds) - 1:
            pm.index += 1
            for fpcd in play_clouds_movement.frame_pcds:
                vis.remove_geometry(fpcd)
            time.sleep(1)
            play_clouds_movement.frame_pcds = copy_pcds(list_of_pcds[pm.index])
            for fpcd in play_clouds_movement.frame_pcds:
                vis.add_geometry(fpcd)
            vis.get_view_control().rotate(0, 500)
            vis.poll_events()
            vis.update_renderer()
        else:
            vis.register_animation_callback(reset_motion)
        return False

    play_clouds_movement.vis = o3d.visualization.Visualizer()
    play_clouds_movement.index = 0
    play_clouds_movement.frame_pcds = copy_pcds(list_of_pcds[0])
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1, origin=[0, 0, 0]
    )
    vis = play_clouds_movement.vis
    vis.create_window()
    for fpcd in play_clouds_movement.frame_pcds:
        vis.add_geometry(fpcd)
    vis.add_geometry(mesh_frame)
    vis.get_view_control().rotate(0, 500)
    vis.register_animation_callback(forward)
    vis.run()
    vis.destroy_window()


def plot_3d_pts(pts, paint=False):
    """Plots 3d points
    Args:
        pts (ndarray): points to plot of shape (n, 3)
    """
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    color = np.zeros(pts.shape[0])
    if paint:
        color = np.arange(pts.shape[0])
    ax.scatter3D(pts[:, 0], pts[:, 1], pts[:, 2], c=color)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()


def plot_2d_pts(pts, paint=True, axis_names=("X", "Y")):
    """Plots 3d points
    Args:
        pts (ndarray): points to plot of shape (n, 2)
    """
    color = np.zeros(pts.shape[0])
    if paint:
        color = np.arange(pts.shape[0])
    plt.scatter(pts[:, 0], pts[:, 1], c=color)
    plt.xlabel(axis_names[0])
    plt.ylabel(axis_names[1])
    plt.show()


def plot_1d_pts(pts, paint=True, axis_name="X"):
    """Plots 3d points
    Args:
        pts (ndarray): points to plot of shape (n, 2)
    """
    color = np.zeros(pts.shape[0])
    if paint:
        color = np.arange(pts.shape[0])
    plt.scatter(np.arange(pts.shape[0]), pts, c=color)
    plt.xlabel(axis_name)
    plt.show()


def plot_relative_egomotion_trajectory(egomotion):
    """Plots relative egomotion trajectory
    Args:
        egomotion (ndarray): relative egomotion (nx4x4) array
    """
    T_acc = np.identity(4)
    poses = np.array([0.0, 0.0, 0.0, 1.0])
    p_0 = np.array([0.0, 0.0, 0.0, 1.0])
    for pose in egomotion:
        T_acc = T_acc.dot(pose)
        p_acc = T_acc.dot(p_0)
        poses = np.vstack((poses, p_acc))
    plot_3d_pts(poses, paint=True)


def depth2gif(depth_path: str, output_path: str) -> None:
    """Converts numpy array depth maps to gif
    Args:
        depth_path: path to the folder with the depth maps
        output_path: path to the gif
    """
    depth_path = Path(depth_path)
    gif_name = str(depth_path.parts[-1]) + ".gif"
    color_map = plt.cm.get_cmap("plasma").reversed()
    imgs = []
    for depth in sorted(depth_path.glob("*")):
        depth = np.load(str(depth))["arr_0"] * 12
        depth = (depth - depth.min()) / (depth.max() - depth.min())
        depth = color_map(depth)
        img = Image.fromarray((depth * 255).astype(np.uint8))
        imgs.append(img)
    img, *imgs = imgs
    img.save(
        fp=gif_name,
        format="GIF",
        append_images=imgs,
        save_all=True,
        duration=500,
        loop=0,
    )
    print("Saved gif to:", gif_name)


def colorize_clouds(clouds: list) -> list:
    """Colorizes point clouds
    Args:
        clouds: point clouds to colorize
    Returns:
        colorized_clouds: painted point clouds
    """
    for i, cloud in enumerate(clouds):
        cloud.paint_uniform_color(COLORS[i])
