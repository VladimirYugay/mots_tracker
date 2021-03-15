""" Base tracker class """
from abc import ABC, abstractmethod

import numpy as np


class BaseTracker(ABC):
    def __init__(self, max_age=1, min_hits=3, dist_threshold=0.3):
        """ Sets key parameters for tracker class """
        self.max_age = max_age
        self.min_hits = min_hits
        self.dist_threshold = dist_threshold
        self.trackers = []
        self.frame_count = 0
        self.representation_size = 0

    def update(self, sample, intrinsics):
        """ updates the state of the tracker """
        detections, info = self.compute_detections(sample, intrinsics)
        self.frame_count += 1
        trackers_predictions = self.predict()
        matched, unmatched_detections, _ = self.associate_detections_to_trackers(
            detections, trackers_predictions
        )
        self.update_matched_trackers(matched, detections, info)
        self.init_unmatched_trackers(unmatched_detections, detections, info)
        return self.filter_trackers()

    def update_matched_trackers(self, matched_pairs, detections, info):
        """ updates matched trackers """
        for (u, v) in matched_pairs:
            self.trackers[v].update(detections[u, :], info[u])

    def predict(self):
        """ gets predictions from current trackers """
        predictions = np.zeros((len(self.trackers), self.representation_size))
        to_del = []
        for t, prediction in enumerate(predictions):
            prediction[:] = np.array(self.trackers[t].predict())
            if np.any(np.isnan(prediction)):
                to_del.append(t)
        predictions = np.ma.compress_rows(np.ma.masked_invalid(predictions))
        for t in reversed(to_del):
            self.trackers.pop(t)
        return predictions

    @abstractmethod
    def compute_detections(self, sample, intrinsics):
        """ computes representation of an object to track """
        pass

    @abstractmethod
    def init_unmatched_trackers(self, unmatched_detections_ids, detections, info):
        """ initializes new trackers with unmatched detections """
        pass

    @abstractmethod
    def filter_trackers(self):
        """ removes trackers which do not match paraameters """
        pass

    @abstractmethod
    def associate_detections_to_trackers(self, detections, trackers):
        pass
