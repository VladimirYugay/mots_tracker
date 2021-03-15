import numpy as np
from filterpy.kalman import KalmanFilter

from mots_tracker.kalman_filters.base_kalman_filter import BaseKalmanFilter


class MedianKalmanFilter(BaseKalmanFilter):
    """ This class represents the internal state of individual tracked objects observed as median point """

    count = 0

    def __init__(self, median, info):
        """Initializes the filter with starting position, info is provided for the sake of submission
        Args:
            median (ndarray): 3d position of the object
            info (dict): dictionary with additional information
        """
        BaseKalmanFilter.__init__(self, median, info)
        self.state_size = 6
        self.kf = KalmanFilter(dim_x=self.state_size, dim_z=self.observation_size)
        self.kf.F = np.eye(self.state_size)
        self.kf.F[[0, 1, 2], [3, 4, 5]] = 1  # linear velocity model
        self.kf.H = np.eye(self.observation_size, self.state_size)
        self.init_uncertainty()
        self.kf.x[: self.observation_size] = median[:, None]
        self.id = MedianKalmanFilter.count
        MedianKalmanFilter.count += 1

    def init_uncertainty(self):
        self.kf.R *= 0.001
        # # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10
        self.kf.P[3:, 3:] *= 100  # higher estimate uncertainty to the velocity
        self.kf.Q[
            3:, 3:
        ] *= 0.01  # lower noise for the velocity, people do not make too abrupt movements

    def predict(self):
        """ Advances the state vector and returns the predicted bounding box estimate """
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.kf.x)
        return self.history[-1][:3][:, 0]
