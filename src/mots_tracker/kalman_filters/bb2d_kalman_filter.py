""" Kalman filter for tracking 2D bounding boxes """
import numpy as np
from filterpy.kalman import KalmanFilter

from mots_tracker.kalman_filters.base_kalman_filter import BaseKalmanFilter


def bb2state(bbox):
    """converts bounding box to state of the filter
    Args:
        bbox (ndarray): box of the top left bottom right format
    Returns:
        ndarray: box of center, area, ratio format
    """
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x, y = bbox[0] + w / 2.0, bbox[1] + h / 2.0
    return np.array([x, y, w * h, w / float(h)]).reshape((4, 1))


def state2bb(state):
    """converts state to the bounding box
    Args:
        state (ndarray): state of the format x, y, area, ratio, v_x, v_y, v_area
    Returns:
        ndarray: box of top left bottom right format
    """
    w = np.sqrt(state[2] * state[3])
    h = state[2] / w
    return np.array(
        [state[0] - w / 2.0, state[1] - h / 2.0, state[0] + w / 2.0, state[1] + h / 2.0]
    ).reshape((1, 4))


class BB2DKalmanFilter(BaseKalmanFilter):
    """ This class represents the internal state of individual tracked objects observed as median point """

    count = 0

    def __init__(self, bb2d, info):
        """Initializes the filter with starting position, info is provided for the sake of submission
        Args:
            bb2d (ndarray): bounding box in top-left bottom-right format
            info (dict): dictionary with additional information
        """
        bb2d_state = bb2state(bb2d)
        BaseKalmanFilter.__init__(self, bb2d_state, info)
        self.state_size = 7  # x, y, scale, ration, v_x, v_y, v_scale
        self.kf = KalmanFilter(dim_x=self.state_size, dim_z=self.observation_size)
        self.kf.F = np.eye(self.state_size)
        self.kf.F[[0, 1, 2], [4, 5, 6]] = 1  # linear velocity model
        self.kf.H = np.eye(self.observation_size, self.state_size)
        self.init_uncertainty()
        self.kf.x[: self.observation_size] = bb2d_state
        self.id = BB2DKalmanFilter.count
        BB2DKalmanFilter.count += 1

    def init_uncertainty(self):
        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

    def predict(self):
        """ Advances the state vector and returns the predicted bounding box estimate """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(state2bb(self.kf.x)[0])
        return self.history[-1]

    def update(self, observation, info):
        """updates the state with the observed 3d position
        Args:
            observation (ndarray): observation
            info (dict): dictionary with additional information
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.info = info
        self.kf.update(bb2state(observation))

    def get_state(self):
        return state2bb(self.kf.x)
