import numpy as np
from scipy.optimize import linear_sum_assignment

from functools import lru_cache


# @lru_cache(maxsize=None)
def compute_iou(tracklet_a, box_b):
    """
    Intersection over union
    :param box_a: Tracklet
    :param box_b: Dict
    :return: iou
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    x_1 = np.maximum(tracklet_a.x1, box_b['x1'])
    y_1 = np.maximum(tracklet_a.y1, box_b['y1'])
    x_2 = np.minimum(tracklet_a.x2, box_b['x2'])
    y_2 = np.minimum(tracklet_a.y2, box_b['y2'])

    if (x_2 - x_1) < 0 or (y_2 - y_1) < 0:
        return 0.0

    # compute the area of intersection rectangle
    intersection_area = (x_2 - x_1 + 1) * (y_2 - y_1 + 1)

    # compute the area of both box_a and box_b
    tracklet_a_area = (tracklet_a.x2 - tracklet_a.x1) * (tracklet_a.y2 - tracklet_a.y1)
    box_b_area = (box_b['x2'] - box_b['x1']) * (box_b['y2'] - box_b['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(tracklet_a_area + box_b_area - intersection_area)

    # return the intersection over union value
    if iou >= 0.0 and iou <= 1.0:
        return iou
    else:
        return 0.0


def compute_assignment_matrix(tracks, detections):
    association_mat = np.zeros((len(tracks), len(detections)))

    for t, track in enumerate(tracks):
        for d, detection in enumerate(detections):
            # TODO memoize this function
            iou = compute_iou(track, detection)
            association_mat[t][d] = iou

    # Solve Hungarian/Munkres algorithm
    return (*linear_sum_assignment(-association_mat), association_mat)


def compute_association(tracks: list, detections: list, iou_threshold=0.3):
    """
    Associates previous tracked detections (tracks) with new detections using the Hungarian algorithm (Munkres).
    The assignment cost in cost-matrix for Munkres is the iou between each track box and the detection box.

    :param tracks: list of previous tracked detections, initially empty
    :param detections: list of new detections
    :return: matches, unmatched trackers, unmatched detections
    """

    matched_tracks, matched_detections, association_mat = compute_assignment_matrix(tracks, detections)

    unmatched_tracks = np.array(list(set(range(len(tracks))).difference(matched_tracks)), dtype=np.int64)
    unmatched_detections = np.array(list(set(range(len(detections))).difference(matched_detections)), dtype=np.int64)

    matches = []

    for match in zip(matched_tracks, matched_detections):
        if association_mat[match[0], match[1]] < iou_threshold:
            unmatched_tracks = np.append(unmatched_tracks, match[0])
            unmatched_detections = np.append(unmatched_detections, match[1])
        else:
            matches.append(np.array(match))

    return matches, unmatched_detections, unmatched_tracks
