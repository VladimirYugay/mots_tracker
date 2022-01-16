import numpy as np
from filterpy.kalman import KalmanFilter

from mots_tracker.kalman_filters.base_kalman_filter import BaseKalmanFilter


class PoseKalmanFilter(BaseKalmanFilter):
    """This class represents the internal state of individual
    tracked objects observed as keypoints"""

    count = 0

    def __init__(self, pose, info):
        """Initializes the filter with starting pose,
           info is provided for the sake of submission
        Args:
            pose (ndarray): 3D or 2D pose of an object
            info (dict): dictionary with additional information
        """
        BaseKalmanFilter.__init__(self, pose, info)
        self.observation_size = pose.flatten().shape[0]
        self.state_size = self.observation_size * 2  # we add velocity
        self.kf = KalmanFilter(dim_x=self.state_size, dim_z=self.observation_size)
        self.kf.F = np.eye(self.state_size)
        # linear velocity model
        self.kf.F[
            np.arange(self.observation_size),
            np.arange(self.observation_size) + self.observation_size,
        ] = 1
        self.kf.H = np.eye(self.observation_size, self.state_size)
        self.init_uncertainty()
        self.kf.x[: self.observation_size] = pose[:, None]
        self.id = PoseKalmanFilter.count
        PoseKalmanFilter.count += 1

    def init_uncertainty(self):
        self.kf.R *= 0.001
        # # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10
        self.kf.P[
            self.observation_size :, self.observation_size :
        ] *= 100  # higher estimate uncertainty to the velocity
        self.kf.Q[
            self.observation_size :, self.observation_size :
        ] *= 0.01  # lower noise for the velocity

    def predict(self):
        """Advances the state vector and returns
        the predicted pose estimate"""
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.kf.x)
        return self.kf.x[: self.observation_size][:, 0]
