""" module with util functions """
import cv2
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


def decode_mask(height: int, width: int, mask_string) -> dict:
    """Decodes mask string to mask  matrix
    Args:
        height: height of the mask
        width: width of the mask
        mask_string: rle encoding in bytes or string format
    Returns:
        mask dictionary in coco format
    """
    if isinstance(mask_string, bytes):
        mask_string = mask_string.decode(encoding="UTF-8")
    return rletools.decode(
        {
            "size": [int(height), int(width)],
            "counts": mask_string.encode(encoding="UTF-8"),
        }
    )


def encode_mask(mask):
    """Encodes binary mask to rle string
    Args:
        mask (ndarray): binary mask
    Returns:
        encoding (dict): dict with 'size' and 'counts' string inside
    """
    mask = mask.astype(np.uint8)
    encoding = rletools.encode(np.asfortranarray(mask))
    encoding["counts"] = encoding["counts"]
    return encoding


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
            if np.unique(np.asarray(cloud.points)[:, -1]).shape[0] < 3:  # flat depth
                _, y, _ = box.get_extent()
                min_bound = box.get_min_bound()
                max_bound = box.get_max_bound()
                # "depth" of a person is approximately 1/6 of height
                min_bound[-1] -= y / 6
                max_bound[-1] += y / 6
                box = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
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


def d2ptcloud(depth, intrinsics, filter_func=None):
    """converts depth image to point cloud
    Args:
        depth (ndarray): depth map
        intrinsics (ndarray): intrinsics matrix
        filter_func (function): filter to apply to the point cloud
    Returns:
        (PointCloud): resulting point cloud
    """
    height, width = depth.shape
    intrinsics = o3d.open3d.camera.PinholeCameraIntrinsic(
        width,
        height,
        fx=intrinsics[0][0],
        fy=intrinsics[1][1],
        cx=intrinsics[0][2],
        cy=intrinsics[1][2],
    )
    pt_cloud = o3d.geometry.PointCloud.create_from_depth_image(
        o3d.geometry.Image(depth), intrinsics
    )
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


def compute_mask_clouds_no_color(depth, masks, intrinsics, filter_func=None):
    """Compute pedestrian point clouds from a sample, Syntactic SUGAR
    Args:
        sample (dict): containing image, depth, masks, intrinsics
        filter_func (function): filter function to apply to the point clouds
    Returns:
        list (o3d.geometry.PointCloud): point clouds
    """
    depth_patches = patch_masks(depth, masks)
    return [
        d2ptcloud(depth_patch, intrinsics, filter_func) for depth_patch in depth_patches
    ]


def fill_masks(masks, kernel_size=5):
    """Fills holes in the binary segmentation masks
    Args:
        masks (ndarray): array of shape (n, h, w)
    Returns:
        filled_masks (ndarray): masks with filled holes
    """
    filled_masks = np.zeros_like(masks)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    for i, mask in enumerate(masks):
        filled_masks[i, ...] = cv2.dilate(mask, kernel, iterations=1)
    return filled_masks


def interpolate_depth(depth: np.ndarray, size: tuple) -> np.ndarray:
    """Interpolates 2D depth image
    Args:
        depth: depth map to interpolate
        size: size to which interpolate the image
    Returns:
        upscaled image
    """
    return cv2.resize(depth, dsize=size, interpolation=cv2.INTER_LANCZOS4)


def numpy2o3d(array: np.ndarray) -> o3d.geometry.PointCloud:
    """Converts numpy array to a 3d point cloud

    Args:
        array: array to convert

    Returns:
        o3d.geometry.PointCloud: resulting point cloud
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(array)
    return pcd
