from filterpy.kalman import KalmanFilter as KF
import numpy as np


class KalmanFilter:
    """
    Kalman filter class, basically just a wrapper class for filterpy kalman filter and initial matrices for Yolov3 bounding boxes
    https://filterpy.readthedocs.io/en/latest/kalman/KalmanFilter.html
    """

    def __init__(self, dt=0.01):
        """
        Resources:
        - https://www.youtube.com/playlist?list=PLX2gX-ftPVXU3oUFNATxGXY90AULiqnWT
        - https://towardsdatascience.com/computer-vision-for-tracking-8220759eee85
        - https://arxiv.org/abs/1602.00763
        - https://github.com/sumitbinnani/Detect-and-Track/blob/master/tracking/kalman_tracker.py
        z measurement vector [x1, y1, w, h] (might change to cx and cy, centerpoints instead of top left coordinates)
        x filter state estimate
        P Estimation error (uncertainty, covariance matrix)  8x8 matrix
        Q Process error (uncertainty/noise)
        R Measurement error (uncertainty) 4X4 matrix
        H Measurement function
        F (A) State transition matrix, 8x8 matrix
        B control transition matrix
        """
        self.kf = KF(dim_x=8, dim_z=4)
        self.kf.H = np.concatenate((np.diag(np.ones(4)), np.zeros((4,4))), axis=1)

        self.kf.P = np.diag(np.ones(8)) * 1000
        self.kf.R = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 10, 0],
            [0, 0, 0, 10]
        ])
        self.kf.Q = np.diag(np.ones(8)) * 0.01

        self.dt = np.ones(4) * dt

        self.kf.F = np.diag(self.dt, 4)
        np.fill_diagonal(self.kf.F, 1)

    def update(self, z):
        self.kf.update(z)

    def predict(self):
        self.kf.predict()
        return self.kf.x[:4].T[0].astype(int)
