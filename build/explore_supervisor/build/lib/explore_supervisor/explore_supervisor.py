'''
Purpose: sometimes m-explore-ros2 doesnt complete properly (e.g., return to the goal or stops prematurely)
This file ensures that when all frontiers have been explored, or a timeout is reached with no movement,
the robot returns to then initial pose recorded at startup 
'''

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
import tf2_ros
import subprocess
import math
import time
import os 

pkg_share = get_package_share_directory('explore_lite')

params_file = os.path.join(
    pkg_share, 'config', 'params.yaml'
)



class ExploreSupervisor(Node):
    def __init__(self):
        super().__init__('explore_supervisor')

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.nav_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')

        self.initial_pose = None
        self.last_position = None
        self.last_motion_time = time.time()

        self.timer = self.create_timer(1.0, self.loop)

        self.explore_proc = None
        self.phase = "init"

    def get_robot_pose(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                'map', 'base_link',
                rclpy.time.Time())
            return tf.transform
        except:
            return None

    def launch_explore(self):
        self.get_logger().info("Launching explore node...")
        self.explore_proc = subprocess.Popen([
            'ros2', 'run', 'explore_lite', 'explore',
            '--ros-args',
            '--params-file', params_file,
            '-p', 'use_sim_time:=True'
            
        ])

    def send_home_goal(self):
        self.get_logger().info("Sending return-home goal")

        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()

        goal.pose.pose = self.initial_pose

        self.nav_client.wait_for_server()
        self.nav_client.send_goal_async(goal)

    def loop(self):
        pose_tf = self.get_robot_pose()
        if pose_tf is None:
            return

        x = pose_tf.translation.x
        y = pose_tf.translation.y

        if self.phase == "init":
            pose = PoseStamped().pose
            pose.position.x = x
            pose.position.y = y
            pose.orientation = pose_tf.rotation

            self.initial_pose = pose
            self.last_position = (x, y)

            self.launch_explore()
            self.phase = "exploring"
            return

        if self.phase == "exploring":
            dist = math.hypot(
                x - self.last_position[0],
                y - self.last_position[1])

            if dist > 0.05:
                self.last_motion_time = time.time()
                self.last_position = (x, y)

            # No motion for 30 sec = done
            if time.time() - self.last_motion_time > 30:
                self.get_logger().info("Exploration complete.")
                self.explore_proc.terminate()
                self.send_home_goal()
                self.phase = "returning"

def main():
    rclpy.init()
    node = ExploreSupervisor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()