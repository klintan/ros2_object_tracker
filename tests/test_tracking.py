import unittest
import numpy as np

import pandas as pd

from object_tracking.tracking import Tracker, Tracklet
from object_tracking.association import compute_iou, compute_association
from tests.utils import yolo_to_coords

class AssociationTest(unittest.TestCase):
    def setUp(self):
        data_path = "data"

        df_tracks = pd.read_csv(f"{data_path}/1478019953689774621.txt", delimiter=" ",
                                names=["category", "x", "y", "width", "height"])
        df_detections = pd.read_csv(f"{data_path}/1478019954186238236.txt", delimiter=" ",
                                    names=["category", "x", "y", "width", "height"])

        self.df_tracks = df_tracks[df_tracks['category'] == 0][["x", "y", "width", "height"]]
        self.df_detections = df_detections[df_detections['category'] == 0][["x", "y", "width", "height"]]

        self.tracks = self.df_tracks.apply(lambda tr: yolo_to_coords(tr.values, (1200, 1920)), axis=1).values
        self.detections = self.df_detections.apply(lambda det: yolo_to_coords(det.values, (1200, 1920)), axis=1).values

    def test_tracklet(self):
        tracklet = Tracklet(track_id=1, **self.tracks[0])

        self.assertEqual(self.tracks[0]['x1'], tracklet.x1)
        self.assertEqual(self.tracks[0]['x2'], tracklet.x2)
        self.assertEqual(self.tracks[0]['y1'], tracklet.y1)
        self.assertEqual(self.tracks[0]['y2'], tracklet.y2)
        self.assertEqual(int(self.df_tracks.values[0][2]*1920), tracklet.get_width())

    def test_iou(self):
        tracklet = Tracklet(track_id=1, **self.tracks[0])
        detection = self.detections[0]

        iou = compute_iou(tracklet, detection)

        np.testing.assert_almost_equal(iou, 0.1812, decimal=3)

    def test_association(self):
        tracklets = []
        for idx, track in enumerate(self.tracks):
            tracklet = Tracklet(track_id=idx, **track)
            tracklets.append(tracklet)

        matches, unmatched_detections, unmatched_tracks = compute_association(tracklets, self.detections)

        self.assertEqual(len(matches) + len(unmatched_tracks), len(self.tracks))
        np.testing.assert_array_equal(matches[0], [1, 1])

    def test_tracking(self):
        tracker = Tracker()

        tracker.track(detections=self.tracks)
        tracker.track(detections=self.detections)

        print(tracker.tracks)
