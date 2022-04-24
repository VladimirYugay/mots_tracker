""" profiling MOTS dataset """
import math
from copy import deepcopy

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from mots_tracker import readers, utils, vis_utils
from mots_tracker.io_utils import get_instance, load_yaml

np.set_printoptions(suppress=True, precision=3)


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


def profile_depth_panoptic(sample):

    # vis_utils.plot_image(sample["depth"], image_type="depth")

    panoptic = sample["panoptic_mask"]
    static_img = sample["image"].copy()
    static_img[panoptic == 0] = 0
    static_depth = sample["depth"].copy()
    static_depth[panoptic == 0] = 0

    sobelx = cv2.Sobel(static_depth, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(static_depth, cv2.CV_64F, 0, 1, ksize=5)
    gradient = abs(sobelx + sobely)
    vis_utils.plot_image(gradient)

    vis_utils.plot_image(static_img)

    static_img[gradient > 5] = 0
    static_depth[gradient > 5] = 0

    vis_utils.plot_image(static_img)

    vis_utils.plot_image(static_depth)

    cloud = utils.rgbd2ptcloud(static_img, static_depth, sample["intrinsics"])
    vis_utils.plot_ptcloud(cloud)


def profile_egomotion():
    import numpy as np

    path = (
        "/home/vy/university/thesis/datasets/MOT16/egomotion_ICP/MOT16-14_egomotion.npy"
    )
    egomotion = np.load(path)
    print(egomotion.shape)
    # vis_utils.plot_relative_egomotion_trajectory(egomotion)


def rotation_matrix_from_vectors(vec1, vec2):
    """Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3)
        which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (
        vec2 / np.linalg.norm(vec2)
    ).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


def get_rotation_matrix(axis, theta):
    """Args:
        axis (list): rotation axis of the form [x, y, z]
        theta (float): rotational angle in radians

    Returns:
        array. Rotation matrix.
    """
    axis = np.asarray(axis)
    theta = np.asarray(theta)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array(
        [
            [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
        ]
    )


def profile_egomotion_PCA():

    path = "/home/vy/university/thesis\
        /datasets/MOT16/egomotion_optical_flow/MOT16-14_egomotion.npy"
    egomotion = np.load(path)

    T_acc = np.identity(4)
    poses = np.array([0.0, 0.0, 0.0, 1.0])
    p_0 = np.array([0.0, 0.0, 0.0, 1.0])
    for pose in egomotion:
        T_acc = T_acc.dot(pose)
        p_acc = T_acc.dot(p_0)
        poses = np.vstack((poses, p_acc))

    # plt.scatter(poses[:, 0], poses[:, 1])
    # plt.show()
    # return

    scaler = StandardScaler()
    scaled_poses = scaler.fit_transform(poses[:, :3])
    # scaled_poses = poses[:, :3]
    poses = poses[:, :3]
    np.set_printoptions(suppress=True, precision=3)
    print(poses.mean(axis=0))
    return
    pca = PCA(n_components=2)
    pca.fit(scaled_poses)

    # const_R = np.array(
    #     [[0.523, -0.332, 0.785][-0.332, 0.769, 0.546][-0.785, -0.546, 0.292]]
    # )

    components = pca.components_
    normal_vector = np.cross(components[0], components[1])
    normal_vector = normal_vector / np.linalg.norm(normal_vector)

    # R = rotation_matrix_from_vectors(normal_vector, np.array([0, 1, 0]))

    # print(R)
    inner = np.inner(normal_vector, np.array([0, 0, 1]))
    norm = np.linalg.norm(normal_vector) * np.linalg.norm(np.array([0, 0, 1]))
    cos = inner / norm
    rad = np.arccos(np.clip(cos, -1.0, 1.0))
    # deg = np.rad2deg(rad)
    axis = np.cross(normal_vector, np.array([0, 0, 1]))
    np.set_printoptions(suppress=True, precision=3)
    R = get_rotation_matrix(-axis, rad)
    print(R)
    return
    normal_vector = normal_vector.dot(R)

    scaled_poses = scaled_poses.dot(R)
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    color = np.arange(scaled_poses.shape[0])
    for pc in pca.components_:
        pc = pc.dot(R)
        start, end = pca.mean_, pca.mean_ + pc
        ax.quiver(start[0], start[1], start[2], end[0], end[1], end[2], length=3)
    ax.quiver(*pca.mean_, *(pca.mean_ + normal_vector), color="red")
    ax.scatter3D(scaled_poses[:, 0], scaled_poses[:, 1], scaled_poses[:, 2], c=color)
    # ax.scatter3D(poses[:, 0], poses[:, 1], poses[:, 2], c=color)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.ylabel("Z")
    plt.show()


def get_gradient(depth):
    sobelx = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=5)
    gradient = abs(sobelx + sobely)
    return gradient


def vector_angle(u, v):
    return np.arccos(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))


def get_cross_prod_mat(pVec_Arr):
    # pVec_Arr shape (3)
    qCross_prod_mat = np.array(
        [
            [0, -pVec_Arr[2], pVec_Arr[1]],
            [pVec_Arr[2], 0, -pVec_Arr[0]],
            [-pVec_Arr[1], pVec_Arr[0], 0],
        ]
    )
    return qCross_prod_mat


def caculate_align_mat(pVec_Arr):
    scale = np.linalg.norm(pVec_Arr)
    pVec_Arr = pVec_Arr / scale
    # must ensure pVec_Arr is also a unit vec.
    z_unit_Arr = np.array([0, 0, 1])
    z_mat = get_cross_prod_mat(z_unit_Arr)

    z_c_vec = np.matmul(z_mat, pVec_Arr)
    z_c_vec_mat = get_cross_prod_mat(z_c_vec)

    if np.dot(z_unit_Arr, pVec_Arr) == -1:
        qTrans_Mat = -np.eye(3, 3)
    elif np.dot(z_unit_Arr, pVec_Arr) == 1:
        qTrans_Mat = np.eye(3, 3)
    else:
        qTrans_Mat = (
            np.eye(3, 3)
            + z_c_vec_mat
            + np.matmul(z_c_vec_mat, z_c_vec_mat) / (1 + np.dot(z_unit_Arr, pVec_Arr))
        )

    qTrans_Mat *= scale
    return qTrans_Mat


def get_arrow(begin, end):
    vec_Arr = np.array(end) - np.array(begin)
    vec_len = np.linalg.norm(vec_Arr)
    mesh_arrow = o3d.geometry.TriangleMesh.create_arrow(
        cone_height=0.2 * vec_len,
        cone_radius=0.06 * vec_len,
        cylinder_height=0.8 * vec_len,
        cylinder_radius=0.04 * vec_len,
    )
    mesh_arrow.paint_uniform_color([1, 0, 1])
    mesh_arrow.compute_vertex_normals()

    mesh_sphere_begin = o3d.geometry.TriangleMesh.create_sphere(
        radius=0.1, resolution=20
    )
    mesh_sphere_begin.translate(begin)
    mesh_sphere_begin.paint_uniform_color([0, 1, 1])
    mesh_sphere_begin.compute_vertex_normals()

    mesh_sphere_end = o3d.geometry.TriangleMesh.create_sphere(radius=0.1, resolution=20)
    mesh_sphere_end.translate(end)
    mesh_sphere_end.paint_uniform_color([0, 1, 1])
    mesh_sphere_end.compute_vertex_normals()
    rot_mat = caculate_align_mat(vec_Arr)
    mesh_arrow.rotate(rot_mat, center=False)
    mesh_arrow.translate(np.array(begin))
    return mesh_arrow


def get_plane_PCA(cloud):
    np.set_printoptions(suppress=True, precision=3)
    points = np.asarray(cloud.points)
    scaler = StandardScaler()
    scaled_points = scaler.fit_transform(points)
    pca = PCA(n_components=2)
    pca.fit(scaled_points)
    components = pca.components_
    normal_vector = np.cross(components[0], components[1])
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    return normal_vector


def get_floor_transform(cloud):
    plane_model, inliers = cloud.segment_plane(
        distance_threshold=0.01, ransac_n=3, num_iterations=1000
    )
    inlier_cloud = cloud.select_down_sample(inliers)
    a, b, c, d = plane_model
    plane_normal = np.array(plane_model[:3])
    xz_normal = np.array([0, 1, 0])
    rotation_angle = utils.compute_angle(xz_normal, plane_normal)
    # define in which direction to rotate the plane
    if c > 0:  # plane looks up
        rotation_angle = -rotation_angle
    R = cloud.get_rotation_matrix_from_xyz(np.array([rotation_angle, 0, 0]))
    tmp = deepcopy(cloud)
    tmp.rotate(R)
    shift_num = np.asanyarray(tmp.points)[:, 1].min()
    t = np.array([0, -shift_num, 0])
    return R, t


# ax + by + cz + d = 0


def profile_floor_alignment(sample):
    # vis_utils.plot_image(sample["depth"])

    img, depth, panoptic = (
        sample["image"].copy(),
        sample["depth"].copy(),
        sample["panoptic_mask"].copy(),
    )
    orig_cloud = utils.rgbd2ptcloud(img, depth, sample["intrinsics"])
    transformed_cloud = deepcopy(orig_cloud)
    transformed_cloud.paint_uniform_color([0, 1, 0])

    mask = np.ones_like(panoptic)
    mask[panoptic == 0] = 0
    mask[depth > 80] = 0
    mask[get_gradient(depth) > 30] = 0

    img[mask == 0] = 0
    depth[mask == 0] = 0

    cloud = utils.rgbd2ptcloud(img, depth, sample["intrinsics"])
    box = list(utils.compute_axis_aligned_bbs([cloud]).values())

    # dummy cloud for the test
    rot_cloud = deepcopy(cloud)
    rot_cloud.paint_uniform_color([0, 1, 0])

    # test rotation
    # theta = np.deg2rad(30)
    # theta = np.deg2rad(-30)
    # R = np.array([[1, 0, 0],
    #     [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])
    # rot_cloud.rotate(R)

    # visualize two clouds before alignment
    tmp = cloud + rot_cloud
    o3d.io.write_point_cloud('before_cloud2.pcd', tmp)
    # vis_utils.plot_ptcloud([cloud, rot_cloud] + box)

    # align
    R, t = get_floor_transform(rot_cloud)
    rot_cloud.rotate(R)
    rot_cloud.translate(t)
    tmp = cloud + rot_cloud
    o3d.io.write_point_cloud('after_cloud2.pcd', tmp)    

    # vis_utils.plot_ptcloud([cloud, rot_cloud] + box)

    # transformed_cloud.rotate(R)
    # transformed_cloud.translate(t)
    # vis_utils.plot_ptcloud([orig_cloud, transformed_cloud])
    # print(R)
    # print(t)

    # visualize the cloud after alignment
    # vis_utils.plot_ptcloud([cloud, rot_cloud])
    # vis_utils.plot_ptcloud([cloud, rot_cloud] + box)


def profile_cloud_vertical_alignment(sample):
    img, depth, panoptic = (
        sample["image"].copy(),
        sample["depth"].copy(),
        sample["panoptic_mask"].copy(),
    )

    mask = np.ones_like(panoptic)
    mask[panoptic == 0] = 0
    mask[depth > 50] = 0

    img[mask == 0] = 0
    depth[mask == 0] = 0

    cloud = utils.rgbd2ptcloud(img, depth, sample["intrinsics"])

    # dummy cloud for the test
    rot_cloud = deepcopy(cloud)
    rot_cloud.paint_uniform_color([0, 1, 0])
    theta = np.deg2rad(1)
    R = np.array(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ]
    )
    rot_cloud.rotate(R)

    source_pts = np.asmatrix(np.asarray(cloud.points))
    target_pts = np.asmatrix(np.asarray(rot_cloud.points))
    T = utils.rigid_transform_3D(source_pts, target_pts, False)

    vis_utils.plot_ptcloud([cloud, rot_cloud])

    print("\n", T)
    rot_cloud.transform(T)
    vis_utils.plot_ptcloud([cloud, rot_cloud])


def profile_cloud_vertical_2d_alignment(sample):
    img, depth, panoptic = (
        sample["image"].copy(),
        sample["depth"].copy(),
        sample["panoptic_mask"].copy(),
    )

    mask = np.ones_like(panoptic)
    mask[panoptic == 0] = 0
    mask[depth > 50] = 0
    img[mask == 0] = 0
    depth[mask == 0] = 0

    cloud = utils.rgbd2ptcloud(img, depth, sample["intrinsics"])
    R_ground = align_cloud_RANSAC(cloud)
    cloud.rotate(R_ground)

    # dummy cloud for the test
    rot_cloud = deepcopy(cloud)
    rot_cloud.paint_uniform_color([0, 1, 0])
    theta = np.deg2rad(30)
    R = np.array(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ]
    )
    print(R)
    rot_cloud.rotate(R)

    source_pts = np.asmatrix(np.asarray(cloud.points)[:, [0, 2]])
    target_pts = np.asmatrix(np.asarray(rot_cloud.points)[:, [0, 2]])

    cloud_2d = o3d.geometry.PointCloud()
    cloud_2d_pts = np.asarray(cloud.points)
    cloud_2d_pts[:, 1] = 5
    cloud_2d.colors = cloud.colors
    cloud_2d.points = o3d.utility.Vector3dVector(cloud_2d_pts)

    cloud_2d_rot = o3d.geometry.PointCloud()
    cloud_2d_rot_pts = np.asarray(rot_cloud.points)
    cloud_2d_rot_pts[:, 1] = 5
    cloud_2d_rot.colors = rot_cloud.colors
    cloud_2d_rot.points = o3d.utility.Vector3dVector(cloud_2d_rot_pts)

    T = utils.rigid_transform_3D(source_pts, target_pts, False)

    T_3d = np.eye(4)
    T_3d[:3, :3] = np.array([[T[0][0], 0, T[0][1]], [0, 1, 0], [T[1][0], 0, T[1][1]]])
    T_3d[0][3] = T[0][2]
    T_3d[2][3] = T[1][2]

    print(T_3d)

    vis_utils.plot_ptcloud([cloud_2d, cloud_2d_rot], False)
    cloud_2d_rot.transform(T_3d)
    vis_utils.plot_ptcloud([cloud_2d, cloud_2d_rot], False)

    return

    vis_utils.plot_ptcloud([cloud, rot_cloud])

    print("\n", T)
    rot_cloud.transform(T)
    vis_utils.plot_ptcloud([cloud, rot_cloud])


def main():
    config_path = "./configs/debug_config_remote.yaml"
    config = load_yaml(config_path)
    reader = get_instance(readers, "reader", config)
    seq_id, frame_id = "MOT16-03", 0
    sample = reader.read_sample(
        seq_id,
        frame_id,
    )
    # print(sample.keys())
    profile_image(sample)
    profile_panoptic(sample)
    # profile_depth(sample)
    # profile_instance_segmentation(sample)
    # profile_scene_cloud(sample)
    # profile_crps(sample)
    # profile_depth_panoptic(sample)
    # profile_egomotion()
    # profile_egomotion_PCA()
    # profile_floor_alignment(sample)
    # profile_cloud_vertical_alignment(sample)
    # profile_cloud_vertical_2d_alignment(sample)


if __name__ == "__main__":
    main()
