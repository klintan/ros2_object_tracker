import rclpy.node
import Node

from darknet_ros_msgs.msg import BoundingBoxes
from object_analytics_msgs.msg import TrackedObjects

from tracking import Tracker


class ObjectTrackingNode(Node):
    def __init__(self):
        super(ObjectTrackingNode, self).__init__("object_tracking_node")
        bbox_topic = self.declare_parameter("bbox_topic", "/bbox").value
        tracker_topic = self.declare_parameter("tracker_topic", "/object_tracks").value

        self.detections_sub = self.create_subscription(BoundingBoxes, bbox_topic, self.tracker_callback, 10)
        self.tracker_pub = self.create_publisher(TrackedObjects, tracker_topic, 10)

    def tracker_callback(self, msg):
        pass


def main(args=None):
    rclpy.init(args=args)

    node = ObjectTrackingNode()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()
