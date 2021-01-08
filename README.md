# ROS2 Object tracking

 Subscribes to a topic from darknet_ros node listening for `darknet_ros_msgs::msg::BoundingBoxes` messages. Simple object tracking using the SORT algorithm.
 
 Publishes TrackedObjects messages (from ros2_object_analytics).
 
## Installation
`colcon build --symlink-install`

## Usage
 `ros2 run ros2_object_tracking object_tracking_node`
 
 
## License
MIT