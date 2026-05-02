#!/usr/bin/env python3
'''
Node 1 of 2: SLAM Exploration

Lifecycle:
    1. Launch Nav2 + SLAM (tb3_simulation_launch.py)
    2. Wait for TF to be ready, record initial pose
    3. Launch explore_lite
    4. Monitor motion — when idle timeout is reached, exploration is complete
    5. Save the map
    6. Return robot to initial pose
    7. Record and save end pose to disk for Node 2 (amcl_and_nav.py)

Usage:
    ros2 run mission_manager slam_and_explore
    When instructed, this node can be Ctrl+C'ed in the terminal 
    and the fixed-map navigation/localization stack can be started
    using the relocalization (then nav) package's launch file 

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

END_POSE_FILE = os.path.expanduser('~/code/robot_sim_nav/src/data/mission_end_pose.yaml')


class SlamAndExplore(Node):

    def __init__(self):
        super().__init__('slam_and_explore')

        # Robot Config / Sim parameters
        self.declare_parameter('scenario', 'sandbox')
        self.declare_parameter('robot', 'waffle')

        self.scenario = self.get_parameter('scenario').value
        self.robot = self.get_parameter('robot').value

        # Explore Parameters
        self.declare_parameter('idle_timeout', 25.0)
        self.declare_parameter('map_file', os.path.expanduser('~/code/robot_sim_nav/src/maps/map'))

        self.idle_timeout = self.get_parameter('idle_timeout').value
        self.map_file = self.get_parameter('map_file').value

        explore_pkg_share = get_package_share_directory('explore_lite')
        self.explore_params_file = os.path.join(
            explore_pkg_share, 'config', 'params.yaml'
        )

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Nav2 action client (used only for return-home goal)
        self.nav_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')

        # State
        self.state = 'INIT'

        # Pose tracking
        self.initial_pose = None
        self.last_pose = None
        self.last_motion_time = time.time()

        # Flag set by nav callbacks when home navigation is resolved
        self.return_complete = False

        # Subprocesses — all launched with start_new_session=True so the
        # entire child process tree can be killed via os.killpg()
        self.nav_slam_proc = None
        self.explore_proc = None

        self.timer = self.create_timer(1.0, self.loop)
        self.get_logger().info('slam_and_explore node started.')

    def get_scenario_config(self):
        if self.scenario == "house":
            return {
                'world': '/path/to/tb3_house.sdf.xacro',
                'x': '0.0',
                'y': '0.0',
                'yaw': '0.0'
            }

        else:   # sandbox default
            return {
                'world': '/path/to/tb3_sandbox.sdf.xacro',
                'x': '-2.0',
                'y': '-0.5',
                'yaw': '0.0'
            }

    # ------------------------------------------------------------------
    # TF helpers
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
    # Process management — use killpg so child processes don't orphan
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
            pass  # already dead

    def launch_slam(self):
        self.get_logger().info('Launching Nav2 + SLAM...')
        self.nav_slam_proc = subprocess.Popen(
            [
                'ros2', 'launch', 'nav2_bringup',
                'tb3_simulation_launch.py',
                'slam:=True',
                'use_sim_time:=True',
                'headless:=False'
            ],
            start_new_session=True
        )

    def stop_slam(self):
        self._kill_proc(self.nav_slam_proc, 'Nav2 + SLAM')
        self.nav_slam_proc = None

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
    # End pose — record current TF pose and write to .yaml for Node 2
    # ------------------------------------------------------------------

    def _record_and_save_end_pose(self):
        pose = self.get_pose()

        if pose is None:
            self.get_logger().error(
                'Could not get TF pose for end pose recording. '
                'Saving zeros — correct manually in mission_end_pose.yaml if needed.'
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
            f'Saved end pose to {END_POSE_FILE}: '
            f'x={data["x"]:.3f}  y={data["y"]:.3f}'
        )
        self.get_logger().info(
            '!!! Exploration complete. '
            'Ctrl+C this node and run navigation with AMCL using the launch file. !!!'
        )
        self.return_complete = True

    # ------------------------------------------------------------------
    # Motion monitoring (idle timeout detection)
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
    # Main state machine
    # ------------------------------------------------------------------

    def loop(self):

        if self.state == 'INIT':
            self.launch_slam()
            self.slam_start_time = time.time()
            self.state = 'WAITING_FOR_SLAM'

        elif self.state == 'WAITING_FOR_SLAM':
            # Give SLAM 10 s to initialise before trying to read TF
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
            # Callbacks handle the transition; just wait here.
            # Once return_complete is set we are done — user can Ctrl+C.
            pass


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