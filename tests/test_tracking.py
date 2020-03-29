import unittest

from object_tracking.tracking import KalmanTracker


class KalmanTrackerTest(unittest.TestCase):
    def test_kalman_matrices(self):
        tracker = KalmanTracker()

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

        self.assertEqual(A, tracker.kf.F)
