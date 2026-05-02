#!/usr/bin/env python3
'''
Node 2 of 2: AMCL Localization + Navigation

Mirrors the working manual pipeline:
    ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py         ← user runs this first
    ros2 launch turtlebot3_navigation2 navigation2.launch.py ...     ← this node launches this
    ros2 run relocalization relocalize_then_nav                      ← this node launches this

Lifecycle:
    1. Read end pose written by slam_and_explore (~/mission_end_pose.yaml)
    2. Launch turtlebot3_navigation2 navigation2.launch.py with the saved map
    3. Wait for Nav2 to be ready (confirmed via TF), then publish initial
       pose to AMCL 3 times, 2 s apart
    4. Launch relocalize_then_nav — handles all subsequent user goals

Prerequisites:
    - turtlebot3_gazebo is still running in another terminal
    - slam_and_explore has completed and been killed with Ctrl+C
    - ~/mission_end_pose.yaml exists on disk

Usage:
    ros2 run <your_pkg> amcl_and_nav
    Then use "Publish Point" in RViz to send navigation goals.
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

END_POSE_FILE = os.path.expanduser('~/mission_end_pose.yaml')


class AmclAndNav(Node):

    def __init__(self):
        super().__init__('amcl_and_nav')

        self.declare_parameter(
            'map_file',
            os.path.expanduser('~/code/robot_sim_nav/src/maps/map.yaml')
        )
        self.map_file = self.get_parameter('map_file').value

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.initial_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped,
            '/initialpose',
            10
        )

        self.nav2_proc = None
        self.relocalize_then_nav_proc = None

        self.state = 'INIT'
        self.end_pose = None

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

    def start_nav2(self):
        '''
        Launch turtlebot3_navigation2 — the same package used in the
        working manual pipeline. This brings up AMCL, controller_server,
        planner_server, and bt_navigator with TB3-specific parameters,
        without spawning Gazebo or a second robot.
        '''
        self.get_logger().info(
            f'Launching turtlebot3_navigation2 with map: {self.map_file}'
        )
        self.nav2_proc = subprocess.Popen(
            [
                'ros2', 'launch', 'turtlebot3_navigation2',
                'navigation2.launch.py',
                'use_sim_time:=True',
                f'map:={self.map_file}',
            ],
            start_new_session=True
        )

    def stop_nav2(self):
        self._kill_proc(self.nav2_proc, 'turtlebot3_navigation2')
        self.nav2_proc = None

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
                'Ensure slam_and_explore completed before starting this node.'
            )
            return False

        with open(END_POSE_FILE, 'r') as f:
            data = yaml.safe_load(f)

        self.end_pose = Pose()
        self.end_pose.position.x    = float(data['x'])
        self.end_pose.position.y    = float(data['y'])
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
    # AMCL initial pose
    # ------------------------------------------------------------------

    def publish_initial_amcl_pose(self):
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.pose = self.end_pose
        msg.pose.covariance[0]  = 0.25    # x variance
        msg.pose.covariance[7]  = 0.25    # y variance
        msg.pose.covariance[35] = 0.0685  # yaw variance

        self.get_logger().info(
            f'Publishing AMCL initial pose '
            f'({self.amcl_pose_publish_count + 1}/3)...'
        )
        self.initial_pose_pub.publish(msg)

    # ------------------------------------------------------------------
    # TF readiness check — confirms Nav2/AMCL is producing transforms
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
                return  # retry each second until file appears
            self.start_nav2()
            self.nav2_start_time = time.time()
            self.state = 'WAITING_FOR_NAV2'

        elif self.state == 'WAITING_FOR_NAV2':
            # Wait at least 15 s for turtlebot3_navigation2 to fully start,
            # then confirm TF is live before publishing the initial pose.
            if time.time() - self.nav2_start_time < 15.0:
                return

            if not self.nav_stack_ready():
                self.get_logger().info(
                    'Nav2 not yet ready (no map->base_link TF), waiting...',
                    throttle_duration_sec=5.0
                )
                return

            # Publish 3 times, 2 s apart, to ensure AMCL receives it
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
                'System ready. '
                'Use the "Publish Point" tool in RViz to send navigation goals.'
            )
            self.state = 'NAV'

        elif self.state == 'NAV':
            pass  # relocalize_then_nav handles everything from here


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
        node.stop_nav2()
        node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()