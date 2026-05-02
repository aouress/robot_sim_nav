#!/usr/bin/env python3
'''
Purpose: Using standard nav2 messages, turn in small circles before navigating to a goal to relocalize the robot. 
Perform this relocalization procedure until the appropriate covariance threshold.
Get goals from clicked points on the map.  

1. Wait for a clicked point from RViz (/publish_point)
2. Drive in small circles to help AMCL relocalize
3. Send NavigateToPose goal to Nav2

ros2 run relocalization relocalize_then_nav
'''

import math
import time
import threading

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import TwistStamped
from geometry_msgs.msg import PoseStamped

from nav2_msgs.action import NavigateToPose

class RelocalizeThenNav(Node):

    def __init__(self):
        super().__init__('relocalize_then_nav')

        # Publisher for robot motion
        self.cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)

        # Subscribe to RViz clicked point
        self.point_sub = self.create_subscription(
            PointStamped,
            '/clicked_point',
            self.point_callback,
            10
        )

        # Nav2 action client
        self.nav_client = ActionClient(
            self,
            NavigateToPose,
            '/navigate_to_pose'
        )

        self.goal_active = False

        self.get_logger().info('Waiting for clicked point on /clicked_point ...')

    # RViz clicked point callback
    def point_callback(self, msg: PointStamped):

        if self.goal_active:
            self.get_logger().warn('Goal already running.')
            return

        self.goal_active = True

        self.get_logger().info(
            f'Received point: x={msg.point.x:.2f}, y={msg.point.y:.2f}'
        )

        thread = threading.Thread(target=self.execute_mission, args=(msg,), daemon=True)
        thread.start()

    
    # Relocalize then nav to complete the mission
    def execute_mission(self, point_msg):

        self.perform_relocalization()

        self.send_nav_goal(point_msg)

    # Relocalization maneuver
    def perform_relocalization(self):

        self.get_logger().info('Starting relocalization circles...')

        twist = TwistStamped()
        twist.twist.linear.x = 0.02
        twist.twist.angular.z = 0.4
        twist.header.frame_id = 'base_link'
        twist.header.stamp = self.get_clock().now().to_msg()

        start = time.time()
        duration = 10.0   # seconds

        while time.time() - start < duration:
            self.cmd_pub.publish(twist)
            time.sleep(0.1)

        # Stop robot
        stop = TwistStamped()
        stop.twist.linear.x = 0.0
        stop.twist.angular.z = 0.0
        stop.header.frame_id = 'base_link'
        stop.header.stamp = self.get_clock().now().to_msg()

        self.cmd_pub.publish(stop)

        self.get_logger().info('Relocalization complete.')

    # Send Nav2 Goal
    def send_nav_goal(self, point_msg):

        self.get_logger().info('Waiting for Nav2 action server...')

        if not self.nav_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('nav2 action server not available, aborting')
            self.goal_active = False
            return 
        
        goal = NavigateToPose.Goal()

        pose = PoseStamped()
        pose.header.frame_id = point_msg.header.frame_id
        pose.header.stamp = self.get_clock().now().to_msg()

        pose.pose.position.x = point_msg.point.x
        pose.pose.position.y = point_msg.point.y
        pose.pose.position.z = 0.0

        # Keep same heading
        pose.pose.orientation.w = 1.0

        goal.pose = pose

        self.get_logger().info('Sending goal to Nav2...')

        future = self.nav_client.send_goal_async(
            goal,
            feedback_callback=self.feedback_callback
        )

        future.add_done_callback(self.goal_response_callback)

    # ---------------------------------------------------
    def goal_response_callback(self, future):

        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected.')
            self.goal_active = False
            return

        self.get_logger().info('Goal accepted.')

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    # ---------------------------------------------------
    def result_callback(self, future):

        self.get_logger().info('Navigation complete.')
        self.goal_active = False

    # ---------------------------------------------------
    def feedback_callback(self, feedback_msg):
        pass

def main(args=None):

    rclpy.init(args=args)

    node = RelocalizeThenNav()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()