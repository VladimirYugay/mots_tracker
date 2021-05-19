""" module with tracker helper functions """
import numpy as np
import open3d as o3d

from mots_tracker.trackers.numba_iou import convert_3dbox_to_8corner, iou3d

# from bbox.bbox2d import BBox2D
# from bbox.bbox3d import BBox3D
# from bbox.box_modes import XYXY


# from bbox.metrics import jaccard_index_2d, jaccard_index_3d


def linear_assignment(cost_matrix):
    """ assigns trajectories based on cost matrix """
    try:
        import lap

        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])
    except ImportError:
        from scipy.optimize import linear_sum_assignment

        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def pairwise_distance(detections, trackers):
    """ Compute pairwise distance matrix between points """
    return np.linalg.norm(detections[:, None, :] - trackers[None, :, :], axis=-1)


def iou3d_matrix(detections, trackers):
    """computes pairwise 3D IoU between detections and existing bounding boxes
    Args:
        detections (ndarray): n bounding boxes as detections
        trackers (ndarray): m bounding boxes as existing tracking targets
    Returns:
        iou3d_matrix (ndarray): nxm matrix with pairwise iou3d
    """
    # boxes are in format: x, y, z, dx, dy, dz ~ w, l, d in o3d
    n, m = detections.shape[0], trackers.shape[0]
    iou_matrix = np.zeros((n, m))
    for i, det in enumerate(detections):
        for j, track in enumerate(trackers):
            det_box = convert_3dbox_to_8corner(det)
            track_box = convert_3dbox_to_8corner(track)
            iou3d_m, iou2d_m = iou3d(det_box, track_box)
            iou_matrix[i][j] = 0.8 * iou3d_m + 0.2 * iou2d_m
    return iou_matrix


def iou_masks(detection_masks, tracker_masks):
    """Computes IoU between two boolean masks
    Args:
        detection_masks (ndarray): boolean mask of shape hxw
        tracker_masks (ndarray): boolean mask of shape hx2
    Returns:
        float: IoU matrix between all the masks
    """
    n, m = detection_masks.shape[0], tracker_masks.shape[0]
    iou_matrix = np.zeros((n, m))
    for i, det_mask in enumerate(detection_masks):
        for j, track_mask in enumerate(tracker_masks):
            intersection = np.logical_and(det_mask, track_mask).sum()
            union = np.logical_or(det_mask, track_mask).sum()
            iou_matrix[i][j] = intersection / union
    return iou_matrix


def iou_batch(bb_test, bb_gt):
    """computes IoU over 2D bounding boxes
    Args:
        bb_test (ndarray): bounding boxes in top left bottom right format
        bb_gt (ndarray): bounding boxes in top left bottom right format
    Returns:
        ndarray: IoU over batches of boxes
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h
    return wh / (
        (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
        + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1])
        - wh
    )


def median_filter(cloud, radius=1):
    """filters outliers in the point clouds based on point deviations from the median point
    Args:
        cloud (o3d.geometry.PointCloud): point cloud to filter
        radius (float): the maximum possible distance from the cloud median
    Returns:
        filtered_cloud (o3d.geometry.PointCloud) filtered point cloud
    """
    filtered_cloud = o3d.geometry.PointCloud()
    pts, colors = np.asarray(cloud.points), np.asarray(cloud.colors)
    median = np.median(pts, axis=0)
    dist = np.linalg.norm(pts - median, axis=1)
    pts, colors = pts[dist <= radius], colors[dist <= radius]
    filtered_cloud.points = o3d.utility.Vector3dVector(pts)
    filtered_cloud.colors = o3d.utility.Vector3dVector(colors)
    return filtered_cloud


def depth_median_filter(cloud, radius=0.3):
    """filters outliers in the point clouds based on depth deviations from median depth
    Args:
        cloud (o3d.geometry.PointCloud): point cloud to filter
        radius (float): the maximum possible distance from the cloud median
    Returns:
        filtered_cloud (o3d.geometry.PointCloud) filtered point cloud
    """
    filtered_cloud = o3d.geometry.PointCloud()
    pts, colors = np.asarray(cloud.points), np.asarray(cloud.colors)
    dist = np.linalg.norm((pts[:, -1] - np.median(pts[:, -1]))[:, None], axis=1)
    pts = pts[dist <= radius]
    filtered_cloud.points = o3d.utility.Vector3dVector(pts)
    if colors.size != 0:
        colors = colors[dist <= radius]
        filtered_cloud.colors = o3d.utility.Vector3dVector(colors)
    return filtered_cloud
