""" base class for Kalman Filters """
from abc import ABC, abstractmethod


class BaseKalmanFilter(ABC):
    @abstractmethod
    def __init__(self, observation, info):
        self.age = 0
        self.history = []
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.info = info
        self.time_since_update = 0
        self.observation_size = observation.shape[0]  # expects one dimensional vector

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def init_uncertainty(self):
        pass

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
        self.kf.update(observation)
