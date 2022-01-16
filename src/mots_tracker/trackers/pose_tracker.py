""" Class tracking medians """
import numpy as np

from mots_tracker.kalman_filters.pose_kalman_filter import PoseKalmanFilter
from mots_tracker.trackers.base_tracker import BaseTracker
from mots_tracker.trackers.tracker_helpers import linear_assignment, pairwise_distance


class PoseTracker(BaseTracker):
    def __init__(
        self,
        max_age=1,
        min_hits=3,
        dist_threshold=0.1,
        use_egomotion=False,
        num_keypoints=22,
        is_3d=True,
    ):
        BaseTracker.__init__(
            self, max_age=max_age, min_hits=min_hits, dist_threshold=dist_threshold
        )
        # x, y, z coordinates of a median
        self.is_3d = is_3d
        self.num_keypoints = num_keypoints
        self.representation_size = num_keypoints * (is_3d + 2)
        self.use_egomotion = use_egomotion

    def compute_detections(self, sample, intrinsics):
        """ computes representations for the point clouds as medians """

        keypoints = sample["keypoints_3d"] if self.is_3d else sample["keypoints_2d"]
        visibility = keypoints[:, :, -1]
        keypoints = keypoints[:, :, :-1]
        keypoints = keypoints.reshape(keypoints.shape[0], -1)
        # if self.use_egomotion:
        #     for cloud in clouds:
        #         cloud.transform(sample["egomotion"])
        if keypoints.shape[0] == 0:
            keypoints = np.empty((0, self.representation_size))
        info = {
            i: {
                "box": sample["boxes"][i],
                "raw_mask": sample["raw_masks"][i],
                "visibility": visibility,
            }
            for i in range(keypoints.shape[0])
        }
        return keypoints, info

    def init_unmatched_trackers(self, unmatched_detections_ids, detections, info):
        for i in unmatched_detections_ids:
            trk = PoseKalmanFilter(detections[i, :], info[i])
            self.trackers.append(trk)

    def filter_trackers(self):
        i, ret = len(self.trackers), []
        for trk in reversed(self.trackers):
            if (trk.time_since_update < 1) and (
                trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            ):
                ret.append((trk.info["raw_mask"], trk.info["box"], trk.id + 1))
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        return ret

    def associate_detections_to_trackers(self, detections, trackers):
        """ Assigns detections to tracked object (both as bounding boxes) """
        if len(trackers) == 0:
            return (
                np.empty((0, 2), dtype=int),
                np.arange(len(detections)),
                np.empty((0, self.representation_size)),
            )

        distance_matrix = pairwise_distance(detections, trackers)
        np.set_printoptions(suppress=True, precision=3)
        print(distance_matrix)

        if min(distance_matrix.shape) > 0:
            a = (distance_matrix < self.dist_threshold).astype(np.int32)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched_indices = np.stack(np.where(a), axis=1)
            else:
                matched_indices = linear_assignment(distance_matrix)
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

        # filter out matched with large distance
        matches = []
        for m in matched_indices:
            if distance_matrix[m[0], m[1]] > self.dist_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))
        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
