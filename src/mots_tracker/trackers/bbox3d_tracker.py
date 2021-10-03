""" Class for tracking 3D bounding boxes """
from functools import partial

import numpy as np

from mots_tracker import utils
from mots_tracker.kalman_filters.bb3d_kalman_filter import BB3DKalmanFilter
from mots_tracker.trackers.base_tracker import BaseTracker
from mots_tracker.trackers.tracker_helpers import (
    depth_median_filter,
    iou3d_matrix,
    linear_assignment,
)
from mots_tracker.utils import compute_axis_aligned_bbs


class BBox3dTracker(BaseTracker):
    def __init__(
        self,
        max_age=1,
        min_hits=3,
        dist_threshold=0.3,
        use_egomotion=False,
        depth_deviation=0.3,
    ):
        BaseTracker.__init__(
            self,
            max_age=max_age,
            min_hits=min_hits,
            dist_threshold=dist_threshold,
        )
        self.representation_size = 6  # x, y, z, d_x, d_y, d_z ~ w, l, d in o3d
        self.use_egomotion = use_egomotion
        self.accumulated_egomotion = np.identity(4)
        self.depth_deviation = depth_deviation

    def compute_detections(self, sample, intrinsics):
        """ computes representations for the objects to track """
        cloud_filter = partial(depth_median_filter, radius=self.depth_deviation)
        clouds = utils.compute_mask_clouds_no_color(
            sample["depth"], sample["masks"], sample["intrinsics"], cloud_filter
        )
        if self.use_egomotion:
            self.accumulated_egomotion = self.accumulated_egomotion.dot(
                sample["egomotion"]
            )
            for cloud in clouds:
                cloud.transform(self.accumulated_egomotion)
        boxes = compute_axis_aligned_bbs(clouds)
        representations, info = [], {}  # for the sake of speedup
        for cloud_id, box in boxes.items():
            if box is not None:
                representations.append(
                    np.concatenate((box.get_center(), box.get_extent()))
                )
                info[cloud_id] = {
                    "box": sample["boxes"][cloud_id],
                    "mask": sample["masks"][cloud_id],
                    "raw_mask": sample["raw_masks"][cloud_id],
                }
        return np.array(representations), info

    def init_unmatched_trackers(self, unmatched_detections_ids, detections, info):
        for i in unmatched_detections_ids:
            trk = BB3DKalmanFilter(detections[i, :], info[i])
            self.trackers.append(trk)

    def filter_trackers(self):
        i, ret = len(self.trackers), []
        for trk in reversed(self.trackers):
            if (trk.time_since_update < 1) and (
                trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            ):
                ret.append(
                    (
                        trk.info["raw_mask"],
                        trk.info["box"],
                        trk.id + 1,
                    )
                )
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        return ret

    def associate_detections_to_trackers(self, detections, trackers):
        """ Assigns detections to tracked object (both represented as bb) """
        if len(trackers) == 0:
            return (
                np.empty((0, 2), dtype=int),
                np.arange(len(detections)),
                np.empty((0, self.representation_size)),
            )

        iou3dmatrix = iou3d_matrix(detections, trackers, mixed=True)

        if min(iou3dmatrix.shape) > 0:
            a = (iou3dmatrix > self.dist_threshold).astype(np.int32)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched_indices = np.stack(np.where(a), axis=1)
            else:
                matched_indices = linear_assignment(-iou3dmatrix)
        else:
            matched_indices = np.empty(shape=(0, 2))
        unmatched_detections = []
        for d, det in enumerate(detections):
            if d not in matched_indices[:, 0]:
                unmatched_detections.append(d)
        unmatched_trackers = []
        for t, trk in enumerate(trackers):
            if t not in matched_indices[:, 1]:
                unmatched_trackers.append(t)

        # filter out matched with low IOU
        matches = []
        for m in matched_indices:
            if iou3dmatrix[m[0], m[1]] < self.dist_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))
        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
