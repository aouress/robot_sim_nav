#!/usr/bin/env python3
'''
SLAM and Exploration

Procedure: 
    1. Launch SLAM Toolbox + Nav2 navigation stack (NO Gazebo — run that separately)
    2. Wait for TF to be ready, record initial pose
    3. Launch explore_lite
    4. Monitor motion — when idle timeout is reached, exploration is complete
    5. Save the map
    6. Return robot to initial pose
    7. Record and save end pose to disk for Node 2 (amcl_and_nav.py)

Usage:
    Terminal 1:  ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
    Terminal 2:  ros2 run mission_manager slam_and_explore

    When "Exploration complete" appears, Ctrl+C Terminal 2,
    then launch amcl_and_nav in a new terminal.

End pose is saved to: ~/mission_end_pose.yaml
'''

import os
import math
import time
import signal
import subprocess
import yaml

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient

import tf2_ros
from tf2_ros import TransformException

from ament_index_python.packages import get_package_share_directory

END_POSE_FILE = os.path.expanduser('~/mission_end_pose.yaml')


class SlamAndExplore(Node):

    def __init__(self):
        super().__init__('slam_and_explore')

        self.declare_parameter('idle_timeout', 25.0)
        self.declare_parameter('map_file', os.path.expanduser('~/code/robot_sim_nav/maps'))

        self.idle_timeout = self.get_parameter('idle_timeout').value
        self.map_file = self.get_parameter('map_file').value

        explore_pkg_share = get_package_share_directory('explore_lite')
        self.explore_params_file = os.path.join(
            explore_pkg_share, 'config', 'params.yaml'
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.nav_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')

        self.state = 'INIT'
        self.initial_pose = None
        self.last_pose = None
        self.last_motion_time = time.time()
        self.return_complete = False

        # Gazebo is NOT launched here — it must already be running from:
        #   ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
        # SLAM and Nav2 navigation are launched as two separate processes
        # so neither one tries to spawn Gazebo or the robot model.
        self.slam_proc = None
        self.nav_proc = None
        self.explore_proc = None

        self.timer = self.create_timer(1.0, self.loop)
        self.get_logger().info('slam_and_explore node started.')

    # ------------------------------------------------------------------
    # TF
    # ------------------------------------------------------------------

    def get_pose(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                'map', 'base_link',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5)
            )
            return tf.transform
        except TransformException:
            self.get_logger().warning(
                'Waiting for TF map->base_link',
                throttle_duration_sec=5.0
            )
            return None

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

    def launch_slam(self):
        # Launch SLAM Toolbox and Nav2 navigation independently.
        # slam_toolbox provides the map->odom TF.
        # navigation_launch.py brings up controller, planner, bt_navigator etc.
        # Neither of these launches Gazebo or spawns a robot.
        self.get_logger().info('Launching SLAM Toolbox...')
        self.slam_proc = subprocess.Popen(
            [
                'ros2', 'launch', 'slam_toolbox',
                'online_async_launch.py',
                'use_sim_time:=True',
            ],
            start_new_session=True
        )

        self.get_logger().info('Launching Nav2 navigation stack...')
        self.nav_proc = subprocess.Popen(
            [
                'ros2', 'launch', 'nav2_bringup',
                'navigation_launch.py',
                'use_sim_time:=True',
            ],
            start_new_session=True
        )

    def stop_slam(self):
        self._kill_proc(self.slam_proc, 'SLAM Toolbox')
        self._kill_proc(self.nav_proc, 'Nav2 navigation')
        self.slam_proc = None
        self.nav_proc = None

    def launch_explore(self):
        self.get_logger().info('Launching explore_lite...')
        self.explore_proc = subprocess.Popen(
            [
                'ros2', 'run', 'explore_lite', 'explore',
                '--ros-args',
                '--params-file', self.explore_params_file,
                '-p', 'use_sim_time:=True',
            ],
            start_new_session=True
        )

    def stop_explore(self):
        self._kill_proc(self.explore_proc, 'explore_lite')
        self.explore_proc = None

    # ------------------------------------------------------------------
    # Map saving
    # ------------------------------------------------------------------

    def save_map(self):
        self.get_logger().info(f'Saving map to {self.map_file}...')
        subprocess.run([
            'ros2', 'run', 'nav2_map_server', 'map_saver_cli',
            '-f', self.map_file
        ])
        self.get_logger().info('Map saved.')

    # ------------------------------------------------------------------
    # Return-home navigation
    # ------------------------------------------------------------------

    def send_home(self):
        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Nav server not available — skipping return home.')
            self._record_and_save_end_pose()
            return

        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose = self.initial_pose

        self.get_logger().info('Sending return-home goal...')
        future = self.nav_client.send_goal_async(goal)
        future.add_done_callback(self._goal_response_cb)

    def _goal_response_cb(self, future):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().error('Return-home goal rejected — saving pose anyway.')
            self._record_and_save_end_pose()
            return
        self.get_logger().info('Return-home goal accepted.')
        handle.get_result_async().add_done_callback(self._goal_result_cb)

    def _goal_result_cb(self, future):
        self.get_logger().info('Robot returned home.')
        self._record_and_save_end_pose()

    # ------------------------------------------------------------------
    # End pose persistence
    # ------------------------------------------------------------------

    def _record_and_save_end_pose(self):
        pose = self.get_pose()

        if pose is None:
            self.get_logger().error(
                'Could not get TF pose. Saving zeros — correct '
                f'{END_POSE_FILE} manually if needed.'
            )
            data = {'x': 0.0, 'y': 0.0, 'qx': 0.0, 'qy': 0.0, 'qz': 0.0, 'qw': 1.0}
        else:
            data = {
                'x':  float(pose.translation.x),
                'y':  float(pose.translation.y),
                'qx': float(pose.rotation.x),
                'qy': float(pose.rotation.y),
                'qz': float(pose.rotation.z),
                'qw': float(pose.rotation.w),
            }

        with open(END_POSE_FILE, 'w') as f:
            yaml.dump(data, f)

        self.get_logger().info(
            f'Saved end pose: x={data["x"]:.3f}  y={data["y"]:.3f}'
        )
        self.get_logger().info(
            '*** Exploration complete. '
            'Ctrl+C this node, then run amcl_and_nav in a new terminal. ***'
        )
        self.return_complete = True

    # ------------------------------------------------------------------
    # Motion monitoring
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    def loop(self):

        if self.state == 'INIT':
            self.launch_slam()
            self.slam_start_time = time.time()
            self.state = 'WAITING_FOR_SLAM'

        elif self.state == 'WAITING_FOR_SLAM':
            if time.time() - self.slam_start_time < 10.0:
                return
            pose = self.get_pose()
            if pose is None:
                return
            self.initial_pose = Pose()
            self.initial_pose.position.x = pose.translation.x
            self.initial_pose.position.y = pose.translation.y
            self.initial_pose.orientation = pose.rotation
            self.launch_explore()
            self.last_motion_time = time.time()
            self.state = 'EXPLORING'
            self.get_logger().info('Initial pose recorded, exploration started.')

        elif self.state == 'EXPLORING':
            self.monitor_motion()
            if time.time() - self.last_motion_time > self.idle_timeout:
                self.get_logger().info(
                    f'No motion for {self.idle_timeout}s — exploration complete.'
                )
                self.state = 'SAVE_MAP'

        elif self.state == 'SAVE_MAP':
            self.stop_explore()
            self.save_map()
            self.state = 'RETURN_HOME'

        elif self.state == 'RETURN_HOME':
            self.send_home()
            self.state = 'RETURN_WAIT'

        elif self.state == 'RETURN_WAIT':
            pass  # waiting for nav callbacks; Ctrl+C when return_complete


def main(args=None):
    rclpy.init(args=args)
    node = SlamAndExplore()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info('Shutting down slam_and_explore.')
        node.stop_explore()
        node.stop_slam()
        node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()