""" profiling gt bb generation from motsynth """
import pickle
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from mots_tracker import readers, utils, vis_utils
from mots_tracker.io_utils import get_instance, load_yaml
from mots_tracker.readers.motsynth_reader import INTRINSICS
from mots_tracker.trackers import numba_iou, tracker_helpers
from mots_tracker.trackers.tracker_helpers import iou_masks


def profile_aligned_bb():
    """ Profile oriented bounding box """
    mesh_box = o3d.geometry.TriangleMesh.create_box(width=1, height=2, depth=3)
    mesh_box.compute_vertex_normals()
    mesh_box.paint_uniform_color([0.1, 0.9, 0.1])
    box = mesh_box.get_axis_aligned_bounding_box()
    print(np.concatenate((box.get_center(), box.get_extent())))
    new_box = numba_iou.convert_3dbox_to_8corner(
        np.concatenate((box.get_center(), box.get_extent()))
    )
    print(numba_iou.box3d_vol(new_box))
    # new_cluod = o3d.geometry.PointCloud()
    # new_cluod.points = o3d.utility.Vector3dVector(new_box)
    # vis_utils.plot_ptcloud([new_cluod, mesh_box])


def profile_iou3d():
    """ Profile iou3d """
    a = o3d.geometry.TriangleMesh.create_box(width=1, height=2, depth=3)
    a.compute_vertex_normals()
    a.paint_uniform_color([0.1, 0.9, 0.1])

    b = o3d.geometry.TriangleMesh.create_box(width=1, height=2, depth=3)
    b.compute_vertex_normals()
    b.paint_uniform_color([0.1, 0.1, 0.9])
    b.translate([0, 0, 1.5])

    a_box = a.get_axis_aligned_bounding_box()
    b_box = b.get_axis_aligned_bounding_box()
    a_conv = numba_iou.convert_3dbox_to_8corner(
        np.concatenate((a_box.get_center(), a_box.get_extent()))
    )
    b_conv = numba_iou.convert_3dbox_to_8corner(
        np.concatenate((b_box.get_center(), b_box.get_extent()))
    )

    numba_iou.iou3d(a_conv, b_conv)
    # vis_utils.plot_ptcloud([a, b])


def profile_mask2clouds(reader, seq_id, frame_id):
    """ Look at cut clouds """
    sample = reader.read_sample(seq_id, frame_id)
    for mask in sample["masks"]:
        cloud = utils.masks2clouds(
            sample["image"], sample["depth"], mask[None, :], sample["intrinsics"]
        )
        tmp_sample = sample.copy()
        tmp_sample["masks"] = mask[None, :]
        old_cloud = utils.compute_mask_clouds(tmp_sample)
        print(np.all(np.asarray(cloud[0].points) == np.asarray(old_cloud[0].points)))
        print(np.asarray(cloud[0].points).shape, np.asarray(old_cloud[0].points).shape)
        # vis_utils.plot_ptcloud(cloud + old_cloud)


def profile_old_clouds(reader, seq_id, frame_id):
    """ Look at cut clouds """
    sample = reader.read_sample(seq_id, frame_id)
    clouds = utils.compute_mask_clouds(
        sample, filter_func=tracker_helpers.depth_median_filter
    )
    vis_utils.plot_ptcloud(clouds)
    # vis_utils.plot_ptcloud(utils.compute_mask_clouds(sample))


def profile_rotations():
    """ Profile oriented bounding box """
    np.set_printoptions(suppress=True, precision=3)
    mesh_box = o3d.geometry.TriangleMesh.create_box(width=1, height=2, depth=3)
    mesh_box.compute_vertex_normals()
    mesh_box.paint_uniform_color([0.1, 0.9, 0.1])
    rot = Rotation.from_rotvec([0, 0, np.deg2rad(90)]).as_matrix()
    mesh_box.rotate(rot)
    box = mesh_box.get_oriented_bounding_box()
    vis_utils.plot_ptcloud([mesh_box, box], True)


def read_depthmap(name, cam_near_clip, cam_far_clip):
    depth = cv2.imread(name)
    depth = np.concatenate(
        (depth, np.zeros_like(depth[:, :, 0:1], dtype=np.uint8)), axis=2
    )
    depth.dtype = np.uint32
    depth = 0.05 * 1000 / depth.astype("float")
    depth = (
        cam_near_clip
        * cam_far_clip
        / (cam_near_clip + depth * (cam_far_clip - cam_near_clip))
    )
    return depth


def profile_gta():
    """ Profile functionality with GTA dataset """
    path = Path("/home/vy/university/thesis/datasets/sample_1")
    info = pickle.load(open(str(path / "info_frames.pickle"), "rb"))
    infot = info[0]
    cam_near_clip = infot["cam_near_clip"]
    cam_far_clip = infot["cam_far_clip"]
    depth = read_depthmap(str(path / "00638.png"), cam_near_clip, cam_far_clip).astype(
        np.float32
    )
    vis_utils.plot_image(depth, image_type="depth")
    image = np.array(utils.load_image(str(path / "00638.jpg")))
    cloud = utils.rgbd2ptcloud(image, depth, INTRINSICS)
    vis_utils.plot_ptcloud(cloud)


def profile_back_proj(sample):
    """ Profile back projection """
    # vis_utils.plot_image_masks(
    #   sample['image'], sample['masks'], sample['mask_ids'] % len(vis_utils.M_COLORS))
    from functools import partial

    cloud_filter = partial(tracker_helpers.depth_median_filter, radius=1)
    clouds = utils.compute_mask_clouds(sample, cloud_filter)
    projections = [
        utils.cloud2img(cloud, (1080, 1920), sample["intrinsics"]) for cloud in clouds
    ]
    vis_utils.plot_image_masks(
        sample["image"], projections, sample["mask_ids"] % len(vis_utils.M_COLORS)
    )
    projections = utils.fill_masks(np.array(projections))
    vis_utils.plot_image_masks(
        sample["image"], projections, sample["mask_ids"] % len(vis_utils.M_COLORS)
    )


def profile_back_proj_transformation(reader, seq_id="045", frame_id=347):
    """ See how reprojection works in tracker iou """
    np.set_printoptions(suppress=True, precision=3)
    sample_left = reader.read_sample(seq_id, frame_id)
    sample_right = reader.read_sample(seq_id, frame_id + 1)
    masks_left = sample_left["masks"][[2, 6, 17, 19], :]
    masks_right = sample_right["masks"][[1, 3, 13, 15], :]

    vis_utils.plot_image_masks(
        sample_left["image"], masks_left, np.arange(masks_left.shape[0])
    )
    vis_utils.plot_image_masks(
        sample_right["image"], masks_right, np.arange(masks_right.shape[0])
    )
    masks_union = np.concatenate((masks_left, masks_right))
    vis_utils.plot_image_masks(sample_right["image"], masks_union, [0, 1, 2, 3] * 2)
    print(iou_masks(masks_left, masks_left))
    print(iou_masks(masks_left, masks_right))

    T_right = sample_right["egomotion"]
    clouds_right = utils.compute_mask_clouds_no_color(
        sample_right["depth"], masks_right, sample_right["intrinsics"]
    )
    for cloud in clouds_right:
        cloud.transform(T_right)

    projections = np.array(
        [
            utils.cloud2img(cloud, (1080, 1920), sample_right["intrinsics"])
            for cloud in clouds_right
        ]
    )
    masks_union = np.concatenate((masks_left, projections))
    print(iou_masks(masks_left, projections))
    vis_utils.plot_image_masks(sample_right["image"], masks_union, [0, 1, 2, 3] * 2)

    projections = utils.fill_masks(projections)
    masks_union = np.concatenate((masks_left, projections))
    print(iou_masks(masks_left, projections))
    vis_utils.plot_image_masks(sample_right["image"], masks_union, [0, 1, 2, 3] * 2)


def profile_filling(sample):
    """ Profile back projection """
    # vis_utils.plot_image_masks(
    #   sample['image'], sample['masks'], sample['mask_ids'] % len(vis_utils.M_COLORS))
    clouds = utils.compute_mask_clouds(sample)
    projections = [
        utils.cloud2img(cloud, (1080, 1920), sample["intrinsics"]) for cloud in clouds
    ]
    projections = np.array(projections)
    vis_utils.plot_image_masks(sample["image"], projections)
    projections = utils.fill_masks(projections)
    vis_utils.plot_image_masks(sample["image"], projections)


def profile_gt_bb(sample):
    cloud_filter = partial(tracker_helpers.depth_median_filter, radius=1)

    vis_utils.plot_image_masks(
        sample["image"], sample["masks"], np.arange(sample["masks"].shape[0])
    )

    clouds = utils.compute_mask_clouds(sample, cloud_filter)
    boxes = utils.compute_axis_aligned_bbs(clouds)
    print([box.volume() for box in boxes.values()])
    vis_utils.plot_ptcloud(clouds + list(boxes.values()))


def main():
    """ visual profiling for generated motsynth bb """
    config = {
        "depth_path": "gt_depth_new",
        "egomotion_path": "egomotion",
        "read_masks": True,
        "read_boxes": True,
        "gt_path": "/home/vy/university/thesis/datasets/MOTSynth_annotations/all",
        "split_path": None,
    }

    config_path = "./configs/median_tracker_config.yaml"
    config = load_yaml(config_path)
    reader = get_instance(readers, "reader", config)

    # 323 too close
    seq_id, frame_id = "045", 348
    sample = reader.read_sample(seq_id, frame_id)
    print(sample.keys())
    # profile_gt_bb(sample)
    # profile_back_proj(sample)
    # profile_gta()
    # profile_back_proj_transformation(reader, seq_id, frame_id)
    # profile_filling(sample)
    # profile_aligned_bb()


if __name__ == "__main__":
    main()
