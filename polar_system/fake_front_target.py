#!/usr/bin/env python3

import math

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from std_msgs.msg import Float64, String
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

class HeadingFrontPointNode(Node):
    """Simple node template:
    - Subscribes to MAVROS local position and compass heading.
    - Converts heading to normalized degrees from north.
    - Computes a point 2 m in front of the drone in local ENU frame.
    """

    def __init__(self):
        super().__init__("heading_front_point_node")

        self.local_pose = None
        self.heading_deg_north = None

        qos_profile_BE = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=8
        )

        self.declare_parameter("topic_pose", "/mavros/local_position/pose")
        self.declare_parameter("topic_heading", "/mavros/global_position/compass_hdg")
        self.declare_parameter("topic_activation", "/polar/activation")
        self.declare_parameter("topic_front_point", "/polar/estimated_center")

        self.topic_pose = self.get_parameter("topic_pose").value
        self.topic_heading = self.get_parameter("topic_heading").value
        self.topic_activation = self.get_parameter("topic_activation").value
        self.topic_front_point = self.get_parameter("topic_front_point").value

        self.declare_parameter("front_point_distance", 5.0)
        self.front_point_distance = self.get_parameter("front_point_distance").value

        self.create_subscription(
            PoseStamped,
            self.topic_pose,
            self.local_pose_callback,
            qos_profile_BE
        )
        self.create_subscription(
            Float64,
            self.topic_heading,
            self.heading_callback,
            qos_profile_BE
        )
        self.create_subscription(
            String,
            self.topic_activation,
            self.activation_callback,
            10,
        )

        self.front_point_pub = self.create_publisher(PoseStamped, self.topic_front_point, 10)
        self.get_logger().info(f"Fake target finished intialization")

    @staticmethod
    def normalize_heading_deg(heading_deg: float) -> float:
        """Normalize heading to [0, 360) degrees from north."""
        return heading_deg % 360.0

    @staticmethod
    def enu_offset_from_heading(heading_deg_north: float, distance_m: float = 2.0):
        """Return (dx, dy) in local ENU given heading in degrees from north.

        ENU convention:
        - x = East
        - y = North
        - heading = 0 deg points North, increases clockwise
        """
        rad = math.radians(heading_deg_north)
        dx = distance_m * math.sin(rad)
        dy = distance_m * math.cos(rad)
        return dx, dy

    def compute_front_point(self, distance_m: float = 2.0):
        """Compute a local position point in front of the drone."""
        if self.local_pose is None or self.heading_deg_north is None:
            return None

        x = self.local_pose.pose.position.x
        y = self.local_pose.pose.position.y
        z = self.local_pose.pose.position.z

        dx, dy = self.enu_offset_from_heading(self.heading_deg_north, distance_m)
        return x + dx, y + dy, z

    def local_pose_callback(self, msg: PoseStamped):
        self.local_pose = msg

    def heading_callback(self, msg: Float64):
        self.heading_deg_north = self.normalize_heading_deg(msg.data)

    def activation_callback(self, msg: String):
        if msg.data != "start":
            return

        front_point = self.compute_front_point(distance_m=self.front_point_distance)
        if front_point is None:
            return

        x, y, z = front_point
        out = PoseStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id  ="map"
        out.pose.position.x = float(x)
        out.pose.position.y = float(y)
        out.pose.position.z = float(z)
        out.pose.orientation = self.local_pose.pose.orientation

        self.front_point_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = HeadingFrontPointNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
