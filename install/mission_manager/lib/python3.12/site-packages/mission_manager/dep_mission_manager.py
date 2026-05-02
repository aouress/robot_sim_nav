#!/usr/bin/env python3

'''
Purpose: This node manages all of the lifecycle changes between state of navgiation operation:

    1. Startup with SLAM and explore the world 
        1.1. sometimes m-explore-ros2 doesnt complete properly (e.g., return to the goal or stops prematurely)
            This file ensures that when all frontiers have been explored, or a timeout is reached with no movement,
            the robot returns to then initial pose recorded at startup   
        1.2. Attempt to return to goal 
    2. Cancel SLAM
    3. Save map created from exploration 
    3. Start AMCL and send initial pose that explore ended at
    4. Run relocalize_then_nav node to improve pose knowledge and then move to clicked point 
'''

import signal
import os
import math
import time
import subprocess

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, PoseWithCovarianceStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient

import tf2_ros
from tf2_ros import TransformException

from ament_index_python.packages import get_package_share_directory

class MissionManager(Node):

    def __init__(self):
        super().__init__('mission_manager')

        # Params
        self.declare_parameter('idle_timeout', 15.0)
        #self.declare_parameter('use_sim_time', True)
        self.declare_parameter('map_file', os.path.expanduser('~/code/robot_sim_nav/maps'))

        explore_pkg_share = get_package_share_directory('explore_lite')
        self.explore_params_file = os.path.join(
            explore_pkg_share, 'config', 'params.yaml'
        ) 

        self.idle_timeout = self.get_parameter('idle_timeout').value
        self.map_file = self.get_parameter('map_file').value

        #  TF for mointoring poses
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Create nav2 client for pose naviation 
        self.nav_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')

        # States 
        self.state = "INIT"
        self.mode = "SLAM"

        # Pose information
        self.initial_pose = None
        self.end_explore_pose = None
        self.last_pose = None
        self.last_motion_time = time.time()
        self.end_pose_reached_flag = False

        # Important subprocesses
        self.explore_proc = None
        self.amcl_proc = None
        self.nav_slam_proc = None
        self.relocalize_then_nav_proc = None

        # publisher intial amcl pose 
        self.initial_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped,
            '/initialpose',
            10
        )

        self.timer = self.create_timer(1.0, self.loop)

        self.get_logger().info("Mission manager started.")

    # TF Pose
    def get_pose(self):
        try:
            tf = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.5))
            return tf.transform
        except TransformException:
            self.get_logger().warning("Waiting for TF map->base", throttle_duration_sec=5.0)
            return None

# STACK LAUNCH       
    # SLAM and Nav2 stack
    def launch_slam(self):
        self.get_logger().info("Launching Nav2 + SLAM...")

        self.nav_slam_proc = subprocess.Popen([
            'ros2', 'launch',
            'nav2_bringup',
            'tb3_simulation_launch.py',
            'slam:=True',
            'use_sim_time:=True'
        ], start_new_session=True)

    def stop_slam(self):
        self.get_logger().info("Stopping SLAM/Nav2...")

        if self.nav_slam_proc:
            try:
                os.killpg(os.getpgid(self.nav_slam_proc.pid), signal.SIGTERM)
                self.nav_slam_proc.wait(timeout=8.0)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(self.nav_slam_proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass  # already dead
            self.nav_slam_proc = None
        
    # Exploration control
    def launch_explore(self):
        self.get_logger().info("Launching exploration...")

        self.explore_proc = subprocess.Popen([
            'ros2', 'run', 'explore_lite', 'explore',
            '--ros-args',
            '--params-file', self.explore_params_file,
            '-p', 'use_sim_time:=True'
        ])

    def stop_explore(self):
        if self.explore_proc:
            self.get_logger().info("Stopping explore node...")
            self.explore_proc.terminate()
            self.explore_proc = None

    def launch_relocalize_then_nav(self):
        self.get_logger().info("Starting navigation with relocalization...")
        self.relocalize_then_nav_proc = subprocess.Popen([
            'ros2', 'run', 'relocalization', 'relocalize_then_nav'
        ])

    def stop_relocalize_then_nav(self):
        if self.relocalize_then_nav_proc:
            self.get_logger().info("Stopping navigation with relocalization...")
            self.relocalize_then_nav_proc.terminate()
            try:
                self.relocalize_then_nav_proc.wait(timeout=5.0)
            except:
                self.relocalize_then_nav_proc.kill()
            self.relocalize_then_nav_proc = None

# END OF STACK LAUNCH

    # Map saving
    def save_map(self):
        self.get_logger().info("Saving map...")

        subprocess.run([
            'ros2', 'run',
            'nav2_map_server',
            'map_saver_cli',
            '-f', self.map_file
        ])

    # Return home (end exploration with SLAM)
    def send_home(self):
        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Nav server not available.")
            return

        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose = self.initial_pose

        self.get_logger().info("Sending return-home goal...")
        future = self.nav_client.send_goal_async(goal)
        future.add_done_callback(
            self.goal_response_callback
        )

    def set_end_explore_pose(self):
        self.end_pose_reached_flag = True
        pose = self.get_pose()
        if pose is None:
            return

        self.end_explore_pose = Pose()
        self.end_explore_pose.position.x = pose.translation.x
        self.end_explore_pose.position.y = pose.translation.y
        self.end_explore_pose.orientation = pose.rotation

    def goal_response_callback(self, future):
        self.nav_goal_handle = future.result()

        if not self.nav_goal_handle.accepted:
            self.get_logger().error(
                "Return-home goal rejected."
            )
            self.stop_slam()
            self.set_end_explore_pose()
            return

        self.get_logger().info(
            "Return-home goal accepted."
        )

        result_future = (
            self.nav_goal_handle.get_result_async()
        )
        result_future.add_done_callback(
            self.goal_result_callback
        )

    def goal_result_callback(self, future):
        self.get_logger().info(
            "Robot returned home successfully after exploration."
        )
        self.stop_slam()
        self.set_end_explore_pose()

    # Start AMCL localization + nav2 stack
    def start_amcl(self):

        self.get_logger().info("Starting AMCL localization...")

        #nav2_pkg = get_package_share_directory('nav2_bringup')
        #tb3_params = os.path.join(nav2_pkg, 'params', 'nav2_params.yaml')

        self.amcl_proc = subprocess.Popen([
            'ros2', 'launch',
            'nav2_bringup',
            'tb3_simulation_launch.py',
            'slam:=False',
            'use_sim_time:=True',
            #'use_robot_state_pub:=True',
            f'map:={self.map_file}.yaml',
            #f'params_file:={tb3_params}'
        ], start_new_session=True)

    # Publish initial pose for AMCL
    def publish_initial_amcl_pose(self):
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()

        msg.pose.pose = self.end_explore_pose

        # simple covariance
        msg.pose.covariance[0] = 0.25
        msg.pose.covariance[7] = 0.25
        msg.pose.covariance[35] = 0.0685

        self.get_logger().info("Publishing AMCL initial pose...")
        self.initial_pose_pub.publish(msg)

    # Check for exploration completion using motion timeouts
    def monitor_motion(self):
        pose = self.get_pose()
        if pose is None:
            return

        x = pose.translation.x
        y = pose.translation.y

        if self.last_pose is None:
            self.last_pose = (x, y)
            return

        dist = math.hypot(x - self.last_pose[0], y - self.last_pose[1])

        if dist > 0.05:
            self.last_motion_time = time.time()
            self.last_pose = (x, y)

    # State machine
    def loop(self):

        # INIT
        if self.state == "INIT":

            # start SLAM and begin exploring
            self.launch_slam()

            self.slam_start_time = time.time()
            self.state = "WAITING_FOR_SLAM"

        elif self.state == "WAITING_FOR_SLAM":
            if( time.time() - self.slam_start_time > 10.0):
                pose = self.get_pose()
                if pose is None:
                    return
                self.initial_pose = Pose()
                self.initial_pose.position.x = pose.translation.x
                self.initial_pose.position.y = pose.translation.y
                self.initial_pose.orientation = pose.rotation

                self.launch_explore()
                self.state = "EXPLORING"
                self.last_motion_time = time.time()

                self.get_logger().info("Initial pose stored, exploring started.")
                return

        # EXPLORING
        elif self.state == "EXPLORING":
            self.monitor_motion()

            if time.time() - self.last_motion_time > self.idle_timeout:
                self.get_logger().info("Exploration complete detected.")
                self.state = "SAVE_MAP"
            return

        # SAVE MAP
        elif self.state == "SAVE_MAP":
            self.save_map()
            self.state = "RETURN_HOME"
            return

        # RETURN HOME (SLAM)
        elif self.state == "RETURN_HOME":
            self.send_home()
            self.stop_explore()
            self.state = "RETURN_WAIT"
            return
        
        # WAITING FOR RETURN TO COMPLETE
        elif self.state == "RETURN_WAIT":
            # if returning to initial pose is complete or its been rejected, start AMCL
            if(self.end_pose_reached_flag == True):
                self.slam_stop_time = time.time()
                self.state = "WAITING_FOR_SLAM_DEATH"
            return 
        
        elif self.state == "WAITING_FOR_SLAM_DEATH":
            if time.time() - self.slam_stop_time > 5.0:
                self.state = "SWITCH_TO_AMCL"
            return

        # SWITCH TO AMCL
        elif self.state == "SWITCH_TO_AMCL":

            self.start_amcl()

            self.state = "WAITING_FOR_AMCL"
            self.amcl_start_time = time.time()

        # WAITING FOR AMCL STARTUP
        elif self.state == "WAITING_FOR_AMCL":
            if(time.time() - self.amcl_start_time > 15.0):

                if not hasattr(self, 'amcl_pose_publish_count'):
                    self.amcl_pose_publish_count = 0
                    self.amcl_last_publish_time = 0.0
                
                if time.time() - self.amcl_last_publish_time > 2.0:
                    self.publish_initial_amcl_pose()
                    self.amcl_pose_publish_count += 1
                    self.amcl_last_publish_time = time.time()

                if self.amcl_pose_publish_count >= 3:
                    self.state = "LOCALIZATION_MODE"
                    self.get_logger().info("Switched to AMCL mode.")
            return

        # LOCALIZATION MODE
        elif self.state == "LOCALIZATION_MODE":
            self.launch_relocalize_then_nav()
            self.get_logger().info("System in AMCL localization mode.")
            self.state = "NAV"
            return
        
        # relocalization package handles remaining logic
        elif self.state == "NAV":
            return 


def main():
    rclpy.init()
    node = MissionManager()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()