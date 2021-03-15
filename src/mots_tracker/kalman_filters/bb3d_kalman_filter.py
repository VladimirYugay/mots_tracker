""" 3D bounding box kalman filter """
import numpy as np
from filterpy.kalman import KalmanFilter

from mots_tracker.kalman_filters.base_kalman_filter import BaseKalmanFilter


class BB3DKalmanFilter(BaseKalmanFilter):
    """ This class represents the internel state of individual tracked objects observed as bbox """

    count = 0

    def __init__(self, bbox3d, info):
        """Initializes the filter with starting position, info is provided for the sake of submission
        Args:
            bbox3d (ndarray): axis aligned bounding box in the format (x, y, z, w, h, l)
            info (dict): dictionary with additional information
        """
        BaseKalmanFilter.__init__(self, bbox3d, info)
        # define constant velocity model x, y, z, w, h, l, v_x, v_y, v_z
        self.state_size = 9
        self.kf = KalmanFilter(dim_x=self.state_size, dim_z=self.observation_size)
        self.kf.F = np.eye(self.state_size)
        self.kf.F[[0, 1, 2], [6, 7, 8]] = 1  # linear velocity model
        self.kf.H = np.eye(self.observation_size, self.state_size)
        self.init_uncertainty()
        self.kf.x[: self.observation_size] = bbox3d[:, None]
        self.id = BB3DKalmanFilter.count
        BB3DKalmanFilter.count += 1

    def init_uncertainty(self):
        pass

    def predict(self):
        """ Advances the state vector and returns the predicted bounding box estimate """
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.kf.x)
        return self.history[-1][: self.observation_size][:, 0]
