import unittest

import pandas as pd

import rclpy
from object_tracking.object_tracking_node import ObjectTrackingNode
from object_tracking.tracking import Tracklet
from darknet_ros_msgs.msg import BoundingBoxes, BoundingBox

from tests.utils import yolo_to_coords


class ObjectTrackingNodeTest(unittest.TestCase):

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

    def test_tracked_object_message(self):
        # Grab first track sample
        track = self.tracks[0]

        track_id = 1
        tr = Tracklet(track_id=track_id, **track)

        tracking_msg = ObjectTrackingNode.tracked_object_message(tr)

        self.assertEqual(tracking_msg.roi.x_offset, track['x1'])

    def test_tracker_callback(self):
        args = None
        rclpy.init(args=args)
        node = ObjectTrackingNode()

        bboxes_msg = BoundingBoxes()

        for track in self.tracks:
            bbox_msg = BoundingBox()

            bbox_msg.xmin = track["x1"]
            bbox_msg.ymin = track["y1"]
            bbox_msg.xmax = track["x2"]
            bbox_msg.ymax = track["y2"]

            bboxes_msg.bounding_boxes.append(bbox_msg)

        node.tracker_callback(bboxes_msg)
