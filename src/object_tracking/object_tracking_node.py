from rclpy.node import Node

from darknet_ros_msgs.msg import BoundingBoxes
from object_analytics_msgs.msg import TrackedObjects, TrackedObject
from sensor_msgs.msg import RegionOfInterest

from object_tracking.tracking import Tracker


class ObjectTrackingNode(Node):
    def __init__(self):
        super(ObjectTrackingNode, self).__init__("object_tracking_node")
        bbox_topic = self.declare_parameter("bbox_topic", "/bbox").value
        tracker_topic = self.declare_parameter("tracker_topic", "/object_tracks").value

        self.detections_sub = self.create_subscription(BoundingBoxes, bbox_topic, self.tracker_callback, 10)
        self.tracker_pub = self.create_publisher(TrackedObjects, tracker_topic, 10)

        self.tracker = Tracker()

    @staticmethod
    def bbox_msg_to_detection(bbox_msg):
        return {"x1": bbox_msg.xmin, "y1": bbox_msg.ymin, "x2": bbox_msg.xmax, "y2": bbox_msg.ymax}

    def tracker_callback(self, msg):
        detections = [ObjectTrackingNode.bbox_msg_to_detection(bbox) for bbox in msg.bounding_boxes]

        tracks = self.tracker.track(detections=detections)

        tracking_msgs = TrackedObjects()

        for track in tracks:
            if track.hits >= 1:
                tracking_msgs.tracked_objects.append(self.tracked_object_message(track))

        tracking_msgs.header.stamp = msg.header.stamp
        tracking_msgs.header.frame_id = msg.header.frame_id
        self.tracker_pub.publish(tracking_msgs)

    @staticmethod
    def tracked_object_message(track):
        msg = TrackedObject()
        msg.id = track.track_id

        roi_msg = RegionOfInterest()
        roi_msg.x_offset = int(track.x1)
        roi_msg.y_offset = int(track.y1)
        roi_msg.height = int(track.get_height())
        roi_msg.width = int(track.get_width())

        msg.roi = roi_msg
        return msg


def main(args=None):
    rclpy.init(args=args)

    node = ObjectTrackingNode()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()
