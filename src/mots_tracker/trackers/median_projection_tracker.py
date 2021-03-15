""" Class tracking medians based on projected IoU """
import numpy as np

from mots_tracker import utils
from mots_tracker.kalman_filters import MedianKalmanFilter
from mots_tracker.trackers import BaseTracker
from mots_tracker.trackers.tracker_helpers import (
    depth_median_filter,
    iou_masks,
    linear_assignment,
)


class MedianProjectionTracker(BaseTracker):
    def __init__(self, max_age=1, min_hits=3, dist_threshold=0.3):
        BaseTracker.__init__(
            self, max_age=max_age, min_hits=min_hits, dist_threshold=dist_threshold
        )
        self.representation_size = 3  # x, y, z coordinates of a median
        self.projections = np.empty((0, 3))

    def update(self, sample, intrinsics):
        """ updates the state of the tracker """
        detections, info = self.compute_detections(
            sample, intrinsics
        )  # (medians, masks, egomotion, intrinsic), info
        self.frame_count += 1
        trackers_predictions = self.predict()  # (medians, velocities, clouds)
        matched, unmatched_detections, _ = self.associate_detections_to_trackers(
            detections, trackers_predictions
        )
        medians, _, _, _ = detections
        self.update_matched_trackers(matched, medians, info)
        self.init_unmatched_trackers(unmatched_detections, medians, info)
        return self.filter_trackers()

    def associate_detections_to_trackers(self, detections, trackers_predictions):
        """ Assigns detections to tracked object (both represented as bounding boxes) """
        if len(trackers_predictions[0]) == 0:
            return (
                np.empty((0, 2), dtype=int),
                np.arange(len(detections[0])),
                np.empty((0, self.representation_size)),
            )

        # detections are (medians + vel, clouds)
        detection_medians, detection_masks, egomotion, intrinsic = detections
        tracker_medians, tracker_velocities, tracker_clouds = trackers_predictions

        # rotate, translate, project tracker point clouds
        dimensions = detection_masks[0].shape
        tracker_translated_clouds = [
            cloud.translate(velocity)
            for cloud, velocity in zip(tracker_clouds, tracker_velocities)
        ]
        tracker_transformed_clouds = [
            cloud.transform(egomotion) for cloud in tracker_translated_clouds
        ]
        projections = np.array(
            [
                utils.cloud2img(cloud, dimensions, intrinsic)
                for cloud in tracker_transformed_clouds
            ]
        )
        self.projections = projections

        iou_matrix = iou_masks(detection_masks, projections)

        if min(iou_matrix.shape) > 0:
            a = (iou_matrix > self.dist_threshold).astype(np.int32)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched_indices = np.stack(np.where(a), axis=1)
            else:
                matched_indices = linear_assignment(-iou_matrix)
        else:
            matched_indices = np.empty(shape=(0, 2))
        unmatched_detections = []
        for d, _ in enumerate(detection_medians):
            if d not in matched_indices[:, 0]:
                unmatched_detections.append(d)
        unmatched_trackers = []
        for t, _ in enumerate(tracker_medians):
            if t not in matched_indices[:, 1]:
                unmatched_trackers.append(t)
        # filter out matched with low IOU
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < self.dist_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))
        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)
        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

    def predict(self):
        """ gets predictions from current trackers """
        predictions = np.zeros(
            (len(self.trackers), 2 * self.representation_size)
        )  # append velocity
        to_del = []
        for t, prediction in enumerate(predictions):
            self.trackers[t].predict()
            prediction[:] = np.array(
                self.trackers[t].history[-1][:, 0]
            )  # get the most recent median and velocity
            if np.any(np.isnan(prediction)):
                to_del.append(t)
        predictions = np.ma.compress_rows(np.ma.masked_invalid(predictions))
        for t in reversed(to_del):
            self.trackers.pop(t)
        medians, velocities = predictions[:, :3], predictions[:, 3:]
        clouds = [tracker.info["cloud"] for tracker in self.trackers]
        return medians, velocities, clouds

    def compute_detections(self, sample, intrinsics):
        """ computes representations for the point clouds as medians """
        clouds = utils.compute_mask_clouds(sample, depth_median_filter)
        medians = np.array(
            [np.median(np.asarray(cld.points), axis=0) for cld in clouds]
        )
        info = {
            i: {"box": sample["boxes"][i], "cloud": clouds[i]}
            for i in range(len(clouds))
        }
        return (
            medians,
            sample["masks"],
            sample["egomotion"],
            sample["intrinsics"],
        ), info

    def init_unmatched_trackers(self, unmatched_detections_ids, detections, info):
        for i in unmatched_detections_ids:
            trk = MedianKalmanFilter(detections[i, :], info[i])
            self.trackers.append(trk)

    def filter_trackers(self):
        i, ret = len(self.trackers), []
        for trk in reversed(self.trackers):
            if (trk.time_since_update < 1) and (
                trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            ):
                ret.append((None, None, trk.info["box"], trk.id + 1))
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        return ret
