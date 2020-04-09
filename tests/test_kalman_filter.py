import unittest
import numpy as np

from object_tracking.tracking import KalmanFilter

class KalmanTrackerTest(unittest.TestCase):
    def test_kalman_matrices(self):
        tracker = KalmanFilter()

        A = np.array([
            [1, 0, 0, 0, 0.01, 0, 0, 0],
            [0, 1, 0, 0, 0, 0.01, 0, 0],
            [0, 0, 1, 0, 0, 0, 0.01, 0],
            [0, 0, 0, 1, 0, 0, 0, 0.01],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ])

        self.assertEqual(A.shape, tracker.kf.F.shape)
        np.testing.assert_array_equal(A, tracker.kf.F)
