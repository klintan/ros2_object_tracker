import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import deque

from object_tracking.kalman_filter import KalmanFilter
from object_tracking.association import compute_association


class Tracklet:
    def __init__(self, track_id, x1, y1, x2, y2, class_id=1):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.w = self.get_width()
        self.h = self.get_height()
        self.track_id = track_id
        self.class_id = class_id

        self.hits = 0
        self.misses = 0

        # create a new kalman filter instance for this track
        self.kf = KalmanFilter()

        # set initial detection bbox as first measurement
        self.kf.update(self.get_z())

    def get_width(self):
        return self.x2 - self.x1

    def get_height(self):
        return self.y2 - self.y1

    def get_bottom_right(self, w, h):
        return self.x1 + w, self.y1 + h

    def get_z(self):
        """
        get measurement vector z for kalman filtering
        """
        return np.array([self.x1, self.y1, self.get_width(), self.get_height()])

    def set_state(self, state):
        self.x1 = state[0]
        self.y1 = state[1]
        self.x2, self.y2 = self.get_bottom_right(state[2], state[3])

    def predict(self):
        # predict the new state
        pred_state = self.kf.predict()

        # update the track with the new predicted state
        self.set_state(pred_state)

    def update_and_predict(self, z):
        # update kalman filter with measurement vector (detection)
        self.kf.update(self.get_z())

        self.predict()


class Tracker:
    def __init__(self, min_hits=1, max_misses=4):
        self.tracks = []
        self.available_ids = deque(list(range(100)))
        self.max_misses = max_misses
        self.min_hits = min_hits

    def track(self, detections):
        matches, unmatched_detections, unmatched_tracks = compute_association(self.tracks, detections)

        for match in matches:
            tracklet = self.tracks[match[0]]
            detection = detections[match[1]]

            tracklet.update_and_predict(detection)

            tracklet.hits += 1
            tracklet.misses = 0

            self.tracks[match[0]] = tracklet

        for unmatched_detection in unmatched_detections:
            tracklet = Tracklet(track_id=self.available_ids.popleft(), **detections[unmatched_detection])
            tracklet.predict()

            # add to tracks list
            self.tracks.append(tracklet)

        for unmatched_track in unmatched_tracks:
            tracklet = self.tracks[unmatched_track]
            tracklet.predict()
            tracklet.misses += 1

            self.tracks[unmatched_track] = tracklet

        # return ids for deleted tracks
        self.available_ids.extend([x.id for x in filter(lambda x: x.misses > self.max_misses, self.tracks)])

        # filter out deleted tracks
        self.tracks = list(filter(lambda x: x.misses < self.max_misses, self.tracks))

        return self.tracks
