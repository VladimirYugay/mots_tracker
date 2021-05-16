""" module with util functions """
import numpy as np
import open3d as o3d
from PIL import Image
from pycocotools import mask as rletools

from mots_tracker import vis_utils


def resize_img(img, img_shape):
    """ resizes an image """
    return img.resize(img_shape, Image.LANCZOS).convert("RGB")


def resize_masks(masks, img_shape):
    """ resizes segmentation masks """
    resized_masks = np.zeros((masks.shape[0], img_shape[1], img_shape[0]))
    for i in range(masks.shape[0]):
        mask_img = Image.fromarray(np.uint8(masks[i] * 255))
        mask_img = mask_img.resize(img_shape, Image.NEAREST)
        mask_img = np.array(mask_img)
        mask_img[mask_img != 0] = 1
        resized_masks[
            i,
        ] = mask_img
    return resized_masks


def resize_boxes(boxes, old_shape, new_shape):
    """ resizes bounding boxes based on old and new shapes of the image """
    boxes = boxes.copy()
    if len(boxes.shape) == 1:
        boxes = boxes[None, :]
    boxes = np.asarray(boxes, dtype=np.float64)
    boxes[:, [0, 2]] *= new_shape[0] / old_shape[0]
    boxes[:, [1, 3]] *= new_shape[1] / old_shape[1]
    # sometimes boxes are outside of the image, we need to clip them
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, new_shape[0] - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, new_shape[1] - 1)
    return boxes


def scale_intrinsics(intrinsics, old_shape, new_shape):
    """Scales intrinsics based on the image size change
    Args:
        intrinsics (ndarray): intrinsics matrix
        old_shape (tuple): old image dimensions  (hxw)
        new_shape (tuple): new image dimensions  (hxw)
    Returns:
        intrinsics (ndarray): rescaled intrinsics
    """
    intrinsics = intrinsics.copy()
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    intrinsics[0, :] *= new_shape[0] / old_shape[0]
    intrinsics[1, :] *= new_shape[1] / old_shape[1]
    return intrinsics


def patch_boxes(image, boxes):
    """Creates image patches with only box regions visible
    Args:
        image (ndarray): depth of RGB image of (h, w, c) or (h, w) shape
        boxes (ndarray): boxes to apply of (n, top_left, bottom_right) format
    Returns:
        patches (ndarray): patches of (n, h, w, c) or (n, h, w) shape
    """
    boxes = boxes.astype(np.int16)
    patches = np.full([boxes.shape[0], *image.shape], 0)
    for i, box in enumerate(boxes):
        width, height = box[2] - box[0], box[3] - box[1]
        patches[
            i,
            box[1] : box[1] + height,
            box[0] : box[0] + width,
        ] = image[box[1] : box[1] + height, box[0] : box[0] + width]
    return patches


def patch_masks(image, masks):
    """Creates image patches with only mask regions visible
    Args:
        image (ndarray): depth or RGB image of (h, w) or (h, w, c) shape
        masks (ndarray): masks to apply of (h, w) shape
    Returns:
        patches (ndarray): patches of (n, h, w, c) or (n, h, w) shape
    """
    patches = np.repeat(image[None, ...], masks.shape[0], axis=0)
    if len(image.shape) == 3:
        masks = np.repeat(masks[..., None], image.shape[2], axis=-1)
    patches[masks == 0] = 0
    return patches


def masks2clouds(image, depth, masks, intrinsics, filter_func=None):
    """creates point clouds corresponding to masks
    Args:
        img (ndarray): rgb image
        depth (ndarray): depth map
        intrinsics (ndarray): intrinsics matrix
        filter_func (function): filter to apply to the point cloud
    Returns:
        pt_clouds (list(PointCloud)): resulting point clouds
    """
    pt_clouds = []
    n_rows, n_cols = depth.shape
    scene = rgbd2ptcloud(image, depth, intrinsics)
    scene_pts = np.asarray(scene.points)
    scene_colors = np.asarray(scene.colors)
    for mask in masks:
        u, v = np.where(mask == 1)
        ids = u * n_cols + v
        pt_cloud = o3d.geometry.PointCloud()
        pt_cloud.points = o3d.utility.Vector3dVector(scene_pts[ids])
        pt_cloud.colors = o3d.utility.Vector3dVector(scene_colors[ids])
        if filter_func is not None:
            pt_cloud = filter_func(pt_cloud)
        pt_clouds.append(pt_cloud)
    return pt_clouds


def decode_mask(height, width, mask_string):
    """ decodes coco segmentation mask string to numpy """
    return rletools.decode(
        {
            "size": [int(height), int(width)],
            "counts": mask_string.encode(encoding="UTF-8"),
        }
    )


def compute_box_center(box):
    """ compute center of the box represented with top left and bottom right corners"""
    return box[0] + (box[2] - box[0]) // 2, box[1] + (box[3] - box[1]) // 2


def compute_axis_aligned_bbs(clouds):
    """Computes axis aligned bounding boxes for the point clouds
        Note: number of boxes might not match number of clouds
        Returning dictionary to handle this possibility
    Args:
        clouds (list(o3d.geometry.PointCloud): point clouds to compute bounding boxes of
    Returns:
        boxes (dict(o3d.geometry.AxisAlignedBoundingBox): axis aligned bounding boxes
    """
    boxes = {i: None for i in range(len(clouds))}
    for i, cloud in enumerate(clouds):
        try:
            box = cloud.get_axis_aligned_bounding_box()
            boxes[i] = box
        except Exception:
            continue
    return boxes


def compute_oriented_bbs(clouds):
    """Computes axis aligned bounding boxes for the point clouds
        Note: number of boxes might not match number of clouds
        Returning dictionary to handle this possibility
    Args:
        clouds (list(o3d.geometry.PointCloud): point clouds to compute bounding boxes of
    Returns:
        boxes (dict(o3d.geometry.AxisAlignedBoundingBox): axis aligned bounding boxes
    """
    boxes = {i: None for i in range(len(clouds))}
    for i, cloud in enumerate(clouds):
        try:
            box = cloud.get_oriented_bounding_box()
            boxes[i] = box
        except Exception:
            continue
    return boxes


def compute_mask_center(mask):
    """ compute center of the binary mask """
    y, x = np.nonzero(mask)
    c_x, c_y = x.min() + (x.max() - x.min()) // 2, y.min() + (y.max() - y.min()) // 2
    return c_x, c_y


def mask_out(img, masks):
    """ masks out objects inside the image or depth """
    img = img.copy()
    combined_mask = masks.sum(axis=0)
    img[combined_mask == 1] = 0
    return img


def load_image(img_file):
    """Load image from disk. Output value range: [0,1]."""
    return Image.open(img_file).convert("RGB")


def rgbd2ptcloud(img, depth, intrinsics, filter_func=None):
    """converts rgbd image to point cloud
    Args:
        img (ndarray): rgb image
        depth (ndarray): depth map
        intrinsics (ndarray): intrinsics matrix
        filter_func (function): filter to apply to the point cloud
    Returns:
        (PointCloud): resulting point cloud
    """
    height, width, _ = img.shape
    rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(img),
        o3d.geometry.Image(depth),
        convert_rgb_to_intensity=False,
        depth_scale=1,
        depth_trunc=1e6,
    )
    intrinsics = o3d.open3d.camera.PinholeCameraIntrinsic(
        width,
        height,
        fx=intrinsics[0][0],
        fy=intrinsics[1][1],
        cx=intrinsics[0][2],
        cy=intrinsics[1][2],
    )
    pt_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_img, intrinsics)
    pt_cloud = pt_cloud if filter_func is None else filter_func(pt_cloud)
    return pt_cloud


def save_img(img, path):
    """saves image array
    Args:
        img (ndarray): img array of h x w x c
        path (str): path to save image to
    """
    if 0 <= img.mean() <= 1.0:
        img = (img * 255).astype(np.uint8)
    img_type = "RGB" if len(img.shape) == 3 else "L"
    img = Image.fromarray(img, mode=img_type)
    img.save(path)


def compute_mask_clouds(sample, filter_func=None, color_weight=None):
    """Compute pedestrian point clouds from a sample, Syntactic SUGAR
    Args:
        sample (dict): containing image, depth, masks, intrinsics
        filter_func (function): filter function to apply to the point clouds
        color_weight (float): color weight to colorize point clouds
    Returns:
        list (o3d.geometry.PointCloud): point clouds
    """
    intrinsics = sample["intrinsics"]
    img_patches = patch_masks(sample["image"], sample["masks"])
    if color_weight is not None:
        img_patches = vis_utils.colorize_patches(
            img_patches, color_weight, sample["box_ids"]
        )
    depth_patches = patch_masks(sample["depth"], sample["masks"])
    return [
        rgbd2ptcloud(img_patch, depth_patch, intrinsics, filter_func)
        for img_patch, depth_patch in zip(img_patches, depth_patches)
    ]


def cloud2img(cloud, dims, intrinsics):
    """Back projects cloud to an image
    Args:
        cloud (o3d.geometry.PointCloud): point cloud to project
        dims (tuple): dimensions to project to
        intrinsics (ndarray): intrinsics matrix
    Returns:
        projected_cloud (ndarray): image with back projected points only
    """
    height, width = dims
    projected_cloud = np.zeros((height, width))
    pts = np.array(cloud.points)
    focal, center = np.eye(3), np.eye(3)
    focal[[0, 1], [0, 1]] = [intrinsics[0][0], intrinsics[1][1]]
    center[[0, 1], [2, 2]] = [intrinsics[0][2], intrinsics[1][2]]
    ppts = center.dot(focal.dot(pts.T)).T
    depth = ppts[:, 2].copy()
    ppts[:, [0, 1]] = ppts[:, [0, 1]] / depth[:, None]
    ppts = np.array(ppts, dtype=np.int)
    valid = (
        (ppts[:, 0] >= 0)
        & (ppts[:, 0] < width)
        & (ppts[:, 1] >= 0)
        & (ppts[:, 1] < height)
    )
    ppts, depth = ppts[valid], depth[valid]
    projected_cloud[ppts[:, 1], ppts[:, 0]] = 1
    return projected_cloud


def rt2transformation(rotation, translation):
    """Converts rotation and translation to transformation in homogenenous coordinates
    Args:
        rotation (ndarray): 3x3 matrix
        translation (ndarry): (3, ) translation vector
    Returns:
        trans (ndarray): (4, 4) transformation matrix
    """
    trans = np.concatenate((rotation, translation[:, None]), axis=1)
    trans = np.concatenate((trans, np.array([0, 0, 0, 1])[None, :]), axis=0)
    return trans
