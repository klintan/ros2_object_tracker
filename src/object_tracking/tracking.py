import numpy as np
from pykalman import KalmanFilter


def iou(box_a, box_b):
    """
    Intersection over union
    :param box_a:
    :param box_b:
    :return: iou
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    x_1 = np.maximum(box_a['x1'], box_b['x1'])
    y_1 = np.maximum(box_a['y1'], box_b['y1'])
    x_2 = np.maximum(box_a['x2'], box_b['x2'])
    y_2 = np.maximum(box_a['y2'], box_b['y2'])

    if (x_2 - x_1) < 0 or (y_2 - y_1) < 0:
        return 0.0

    # compute the area of intersection rectangle
    intersection_area = (x_2 - x_1 + 1) * (y_2 - y_1 + 1)

    # compute the area of both box_a and box_b
    box_a_area = (box_a['x2'] - box_a['x1']) * (box_a['y2'] - box_a['y1'])
    box_b_area = (box_b['x2'] - box_b['x1']) * (box_b['y2'] - box_b['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(box_a_area + box_b_area - intersection_area)

    # return the intersection over union value
    if iou >= 0.0 and iou <= 1.0:
        return iou
    else:
        return 0.0


class KalmanTracker:
    """
    Kalman tracker class
    """

    def __init__(self):
        """
        Resources:
        - https://www.youtube.com/playlist?list=PLX2gX-ftPVXU3oUFNATxGXY90AULiqnWT
        - https://towardsdatascience.com/computer-vision-for-tracking-8220759eee85
        - https://arxiv.org/abs/1602.00763
        - https://github.com/sumitbinnani/Detect-and-Track/blob/master/tracking/kalman_tracker.py
        Estimation error (uncertainty) (P)  8x8 matrix
        Measurement error (uncertainty) (R)
        """
        self.estimation_error = np.diag(np.ones(8)) * 10
        self.measurement_error = np.array([[1, 0, 0, 0],
                                      [0, 1, 0, 0],
                                      [0, 0, 10, 0],
                                      [0, 0, 0, 10]])
        

    def update(self):
        pass

    def predict(self):
        pass
