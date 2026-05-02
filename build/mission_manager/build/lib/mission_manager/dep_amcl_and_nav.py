#!/usr/bin/env python3
'''
Node 2 of 2: AMCL Localization + Navigation

Lifecycle:
    1. Read end pose written by slam_and_explore (~/mission_end_pose.yaml)
    2. Launch full Nav2 stack with AMCL (bringup_launch.py, slam:=False)
    3. Wait for AMCL to be ready, then publish initial pose 3x spaced 2s apart
    4. Launch relocalize_then_nav — handles all subsequent user goals

Prerequisites:
    - slam_and_explore has completed and been killed with Ctrl+C
    - ~/mission_end_pose.yaml exists on disk
    - Gazebo / robot_state_publisher are still running from the previous session
      (tb3_simulation_launch.py keeps Gazebo alive until you kill that terminal)

Usage:
    ros2 run <your_pkg> amcl_and_nav

    Then in RViz, use the "Publish Point" tool (not "2D Pose Goal") to send
    navigation goals via relocalize_then_nav.
'''

import os
import time
import signal
import subprocess
import yaml

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, PoseWithCovarianceStamped

import tf2_ros
from tf2_ros import TransformException

from ament_index_python.packages import get_package_share_directory

END_POSE_FILE = os.path.expanduser('~/code/robot_sim_nav/src/data/mission_end_pose.yaml')


class AmclAndNav(Node):

    def __init__(self):
        super().__init__('amcl_and_nav')

        self.declare_parameter(
            'map_file',
            os.path.expanduser('~/code/robot_sim_nav/src/maps')
        )
        self.map_file = self.get_parameter('map_file').value

        # TF — used only to confirm the nav stack is alive
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Publisher for AMCL initial pose
        self.initial_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped,
            '/initialpose',
            10
        )

        # Subprocesses
        self.amcl_proc = None
        self.relocalize_then_nav_proc = None

        # State
        self.state = 'INIT'
        self.end_pose = None

        # Pose publish bookkeeping
        self.amcl_pose_publish_count = 0
        self.amcl_last_publish_time = 0.0

        self.timer = self.create_timer(1.0, self.loop)
        self.get_logger().info('amcl_and_nav node started.')

    # ------------------------------------------------------------------
    # Process management
    # ------------------------------------------------------------------

    def _kill_proc(self, proc, name):
        if proc is None:
            return
        self.get_logger().info(f'Stopping {name}...')
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=8.0)
        except subprocess.TimeoutExpired:
            self.get_logger().warning(f'{name} did not stop cleanly, sending SIGKILL')
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        except ProcessLookupError:
            pass

    def start_amcl(self):
        self.get_logger().info('Launching Nav2 + AMCL...')

        # Use the same TB3 params that tb3_simulation_launch.py uses so the
        # controller / planner are correctly configured for the TurtleBot3.
        nav2_pkg = get_package_share_directory('nav2_bringup')
        tb3_params = os.path.join(nav2_pkg, 'params', 'nav2_params.yaml')

        self.amcl_proc = subprocess.Popen(
            [
                'ros2', 'launch', 'nav2_bringup',
                'tb3_simulation_launch.py',
                'slam:=False',
                'use_sim_time:=True',
                #'use_robot_state_pub:=True',
                f'map:={self.map_file}.yaml'
                #f'params_file:={tb3_params}',
            ],
            start_new_session=True
        )

    def stop_amcl(self):
        self._kill_proc(self.amcl_proc, 'Nav2 + AMCL')
        self.amcl_proc = None

    def launch_relocalize_then_nav(self):
        self.get_logger().info('Launching relocalize_then_nav...')
        self.relocalize_then_nav_proc = subprocess.Popen(
            ['ros2', 'run', 'relocalization', 'relocalize_then_nav'],
            start_new_session=True
        )

    def stop_relocalize_then_nav(self):
        self._kill_proc(self.relocalize_then_nav_proc, 'relocalize_then_nav')
        self.relocalize_then_nav_proc = None

    # ------------------------------------------------------------------
    # End-pose loading
    # ------------------------------------------------------------------

    def load_end_pose(self):
        if not os.path.exists(END_POSE_FILE):
            self.get_logger().error(
                f'End pose file not found: {END_POSE_FILE}\n'
                'Make sure slam_and_explore completed successfully before '
                'starting this node.'
            )
            return False

        with open(END_POSE_FILE, 'r') as f:
            data = yaml.safe_load(f)

        self.end_pose = Pose()
        self.end_pose.position.x = float(data['x'])
        self.end_pose.position.y = float(data['y'])
        self.end_pose.orientation.x = float(data['qx'])
        self.end_pose.orientation.y = float(data['qy'])
        self.end_pose.orientation.z = float(data['qz'])
        self.end_pose.orientation.w = float(data['qw'])

        self.get_logger().info(
            f'Loaded end pose: x={self.end_pose.position.x:.3f}  '
            f'y={self.end_pose.position.y:.3f}'
        )
        return True

    # ------------------------------------------------------------------
    # AMCL initial pose publication
    # ------------------------------------------------------------------

    def publish_initial_amcl_pose(self):
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.pose = self.end_pose

        # Covariance: x, y, yaw variances in the diagonal
        msg.pose.covariance[0]  = 0.25    # x
        msg.pose.covariance[7]  = 0.25    # y
        msg.pose.covariance[35] = 0.0685  # yaw

        self.get_logger().info(
            f'Publishing AMCL initial pose '
            f'({self.amcl_pose_publish_count + 1}/3)...'
        )
        self.initial_pose_pub.publish(msg)

    # ------------------------------------------------------------------
    # TF check — confirms nav stack is producing transforms
    # ------------------------------------------------------------------

    def nav_stack_ready(self):
        try:
            self.tf_buffer.lookup_transform(
                'map', 'base_link',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5)
            )
            return True
        except TransformException:
            return False

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    def loop(self):

        if self.state == 'INIT':
            if not self.load_end_pose():
                # File missing — keep retrying once per second until the
                # user copies the file or re-runs slam_and_explore
                return
            self.start_amcl()
            self.amcl_start_time = time.time()
            self.state = 'WAITING_FOR_AMCL'

        elif self.state == 'WAITING_FOR_AMCL':
            # Wait at least 15 s for bringup, then confirm TF is live
            # before publishing the initial pose
            elapsed = time.time() - self.amcl_start_time
            if elapsed < 15.0:
                return

            if not self.nav_stack_ready():
                self.get_logger().info(
                    'Nav2 not yet ready (no map->base_link TF), waiting...',
                    throttle_duration_sec=5.0
                )
                return

            # Publish 3 times, 2 s apart
            now = time.time()
            if now - self.amcl_last_publish_time >= 2.0:
                self.publish_initial_amcl_pose()
                self.amcl_pose_publish_count += 1
                self.amcl_last_publish_time = now

            if self.amcl_pose_publish_count >= 3:
                self.get_logger().info('AMCL initialised.')
                self.state = 'LAUNCH_NAV'

        elif self.state == 'LAUNCH_NAV':
            self.launch_relocalize_then_nav()
            self.get_logger().info(
                'System ready. Use the "Publish Point" tool in RViz to send goals.'
            )
            self.state = 'NAV'

        elif self.state == 'NAV':
            # relocalize_then_nav handles everything from here
            pass


def main(args=None):
    rclpy.init(args=args)
    node = AmclAndNav()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info('Shutting down amcl_and_nav.')
        node.stop_relocalize_then_nav()
        node.stop_amcl()
        node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()