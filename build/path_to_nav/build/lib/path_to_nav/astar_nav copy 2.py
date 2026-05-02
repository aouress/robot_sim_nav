#!/usr/bin/env python3
"""
astar_nav.py  —  A* Planner + Relocalization for TurtleBot3 (ROS2 / Nav2)
==========================================================================
Flow
----
1. Wait for /clicked_point from RViz ("Publish Point" tool)
2. Spin in place to help AMCL relocalize until covariance is low enough
3. Run A* on /global_costmap/costmap (already inflated by Nav2's costmap layers)
4. Stream explored nodes → RViz (MarkerArray, blue→green gradient)
5. Publish final path → /astar/path  (Path)
6. Hand path to Nav2's FollowPath action — Nav2's own controller drives the robot

ros2 run <your_package> astar_nav
"""

import math
import heapq
import time
import threading

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
import rclpy.time

from nav_msgs.msg import OccupancyGrid, Path
from nav2_msgs.action import FollowPath
from geometry_msgs.msg import (
    PointStamped, PoseStamped, PoseWithCovarianceStamped, TwistStamped
)
from visualization_msgs.msg import Marker, MarkerArray
import tf2_ros
from tf2_ros import TransformException


# ─────────────────────────────────────────────────────────────────────────────
# Tunable constants  (override with ROS params if desired)
# ─────────────────────────────────────────────────────────────────────────────
OBSTACLE_THRESHOLD   = 65     # costmap value (0-254) above which = obstacle
                               # 0=free, 99=inscribed, 254=lethal
RELOC_LINEAR         = 0.02   # m/s  forward speed during relocalization
RELOC_ANGULAR        = 0.45   # rad/s angular speed during relocalization
RELOC_COV_THRESHOLD  = 0.08   # sum of x+y diagonal covariance → "localized"
RELOC_TIMEOUT        = 20.0   # s — give up waiting for covariance to drop


class AStarNav(Node):

    def __init__(self):
        super().__init__('astar_nav')

        # ── State ─────────────────────────────────────────────────────────────
        self.costmap: OccupancyGrid | None = None
        self.costmap_lock = threading.Lock()

        self.amcl_cov_xy = float('inf')   # latest x+y position covariance
        self.goal_active  = False

        # ── TF ────────────────────────────────────────────────────────────────
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ── Subscribers ───────────────────────────────────────────────────────
        # Global costmap — already inflated by Nav2's costmap layers, so no
        # manual dilation needed before running A*.
        self.create_subscription(
            OccupancyGrid, '/global_costmap/costmap', self._costmap_cb, 10
        )
        # AMCL pose gives us localization confidence via covariance
        self.create_subscription(
            PoseWithCovarianceStamped, '/amcl_pose', self._amcl_cb, 10
        )
        # Goal input from RViz "Publish Point" tool
        self.create_subscription(
            PointStamped, '/clicked_point', self._goal_cb, 10
        )

        # ── Publishers ────────────────────────────────────────────────────────
        # TRANSIENT_LOCAL: RViz receives the last message even if it subscribes
        # after the node has already published (critical for one-shot viz topics).
        latched = QoSProfile(
            depth=500,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.cmd_pub      = self.create_publisher(TwistStamped,  '/cmd_vel',        10)
        self.path_pub     = self.create_publisher(Path,          '/astar/path',     latched)
        self.explored_pub = self.create_publisher(MarkerArray,   '/astar/explored', latched)

        # ── Nav2 FollowPath action ────────────────────────────────────────────
        # FollowPath lets us supply our own path while Nav2's controller
        # (DWB / RegulatedPurePursuit) handles execution + obstacle avoidance.
        self.follow_path_client = ActionClient(self, FollowPath, '/follow_path')

        self.get_logger().info(
            'A* Nav node ready.\n'
            '  Click a point in RViz (Publish Point → /clicked_point) to start.\n'
            '  Add to RViz:\n'
            '    • MarkerArray  /astar/explored\n'
            '    • Path         /astar/path'
        )

    # =========================================================================
    # SUBSCRIBERS
    # =========================================================================

    def _costmap_cb(self, msg: OccupancyGrid):
        with self.costmap_lock:
            self.costmap = msg

    def _amcl_cb(self, msg: PoseWithCovarianceStamped):
        # Covariance matrix is 6×6 row-major; [0]=xx, [7]=yy
        cov = msg.pose.covariance
        self.amcl_cov_xy = cov[0] + cov[7]   # position uncertainty (x+y)

    def _goal_cb(self, msg: PointStamped):
        if self.goal_active:
            self.get_logger().warn('Goal already active — ignoring new point.')
            return
        self.goal_active = True
        # Run everything in a thread so we never block the ROS executor
        threading.Thread(
            target=self._execute_mission, args=(msg,), daemon=True
        ).start()

    # MISSION  (runs in background thread)

    def _execute_mission(self, point_msg: PointStamped):
        try:
            self._relocalize()
            self._plan_and_navigate(point_msg)
        finally:
            self.goal_active = False

    # STEP 1 — RELOCALIZATION

    def _relocalize(self):
        """
        Spin slowly until AMCL's position covariance drops below
        RELOC_COV_THRESHOLD (robot is well-localized) or RELOC_TIMEOUT expires.
        """
        self.get_logger().info(
            f'Starting relocalization  '
            f'(target covariance ≤ {RELOC_COV_THRESHOLD:.3f}, '
            f'timeout {RELOC_TIMEOUT:.0f} s)...'
        )

        twist = TwistStamped()
        twist.header.frame_id   = 'base_link'
        twist.twist.linear.x    = RELOC_LINEAR
        twist.twist.angular.z   = RELOC_ANGULAR

        start = time.time()
        while time.time() - start < RELOC_TIMEOUT:
            cov = self.amcl_cov_xy
            self.get_logger().info(
                f'  Covariance x+y = {cov:.4f}  '
                f'(threshold {RELOC_COV_THRESHOLD:.3f})',
                throttle_duration_sec=2.0
            )
            if cov <= RELOC_COV_THRESHOLD:
                self.get_logger().info(
                    f'✓ Relocalized!  covariance = {cov:.4f}'
                )
                break
            twist.header.stamp = self.get_clock().now().to_msg()
            self.cmd_pub.publish(twist)
            time.sleep(0.1)
        else:
            self.get_logger().warn(
                f'Relocalization timed out after {RELOC_TIMEOUT:.0f} s '
                f'(final covariance = {self.amcl_cov_xy:.4f}). Proceeding anyway.'
            )

        self._stop_robot()

    # =========================================================================
    # STEP 2 — PLAN WITH A* THEN FOLLOW VIA NAV2
    # =========================================================================

    def _plan_and_navigate(self, point_msg: PointStamped):

        with self.costmap_lock:
            if self.costmap is None:
                self.get_logger().error('No costmap received yet — aborting.')
                return
            costmap = self.costmap   # snapshot (OccupancyGrid is immutable here)

        goal_world = (point_msg.point.x, point_msg.point.y)

        # Robot pose from TF
        try:
            tf = self.tf_buffer.lookup_transform(
                'map', 'base_footprint', rclpy.time.Time()
            )
            start_world = (
                tf.transform.translation.x,
                tf.transform.translation.y,
            )
        except TransformException as e:
            self.get_logger().error(f'TF lookup failed: {e}')
            return

        self.get_logger().info(
            f'\n{"─"*56}\n'
            f'  Start : ({start_world[0]:.2f}, {start_world[1]:.2f})\n'
            f'  Goal  : ({goal_world[0]:.2f},  {goal_world[1]:.2f})\n'
            f'{"─"*56}'
        )

        info       = costmap.info
        start_cell = self._world_to_cell(start_world, info)
        goal_cell  = self._world_to_cell(goal_world,  info)

        # Build binary grid from the global costmap.
        # Costmap values: 0=free, 1-252=cost, 253=inscribed, 254=lethal, -1=unknown.
        # Inflation is already baked in by Nav2 — no extra dilation needed.
        # Unknown cells (-1) treated as free.
        raw  = np.array(costmap.data, dtype=np.int8).reshape(
            (info.height, info.width)
        )
        grid = (raw >= OBSTACLE_THRESHOLD).astype(np.uint8)

        # Bounds check
        H, W = grid.shape
        for label, (r, c) in [('Start', start_cell), ('Goal', goal_cell)]:
            if not (0 <= r < H and 0 <= c < W):
                self.get_logger().error(f'{label} ({r},{c}) is outside costmap bounds.')
                return

        # ── A* ────────────────────────────────────────────────────────────────
        self.get_logger().info('Running A*...')
        t0 = time.time()
        path_cells, explored = self._astar(grid, start_cell, goal_cell)
        elapsed_ms = (time.time() - t0) * 1000

        if path_cells is None:
            self.get_logger().error(
                f'A* found no path after exploring {len(explored)} nodes.'
            )
            self._publish_explored(explored, info)   # still show what was searched
            return

        self.get_logger().info(
            f'\n  A* complete\n'
            f'  ├─ Nodes explored : {len(explored)}\n'
            f'  ├─ Path length    : {len(path_cells)} cells\n'
            f'  └─ Planning time  : {elapsed_ms:.1f} ms'
        )

        # ── Visualise ─────────────────────────────────────────────────────────
        path_world = [self._cell_to_world(c, info) for c in path_cells]
        self._publish_explored(explored, info)    # animated search spread
        nav2_path  = self._build_path_msg(path_world, costmap.header.frame_id)
        self.path_pub.publish(nav2_path)          # final path for RViz
        self.get_logger().info('✓ Path published to /astar/path')

        # ── Hand to Nav2 ──────────────────────────────────────────────────────
        self._follow_path(nav2_path)

    # =========================================================================
    # A* ALGORITHM
    # =========================================================================

    def _astar(
        self,
        grid:  'np.ndarray',
        start: tuple[int, int],
        goal:  tuple[int, int],
    ) -> tuple[list | None, list]:
        """
        8-connected A* on a binary occupancy grid.

        Returns
        -------
        path_cells : list[(row, col)] start→goal, or None if unreachable
        explored   : list[(row, col)] in expansion order  ← shown in RViz
        """
        sr, sc = start
        gr, gc = goal
        H, W   = grid.shape

        def h(r, c):                         # Euclidean heuristic (admissible)
            return math.hypot(r - gr, c - gc)

        # (Δr, Δc, move cost)  — diagonal cost = √2 for geometric accuracy
        MOVES = [
            (-1,  0, 1.0),   ( 1,  0, 1.0),
            ( 0, -1, 1.0),   ( 0,  1, 1.0),
            (-1, -1, 1.414), (-1,  1, 1.414),
            ( 1, -1, 1.414), ( 1,  1, 1.414),
        ]

        open_heap = [(h(sr, sc), 0.0, sr, sc)]
        g_score   = {(sr, sc): 0.0}
        came_from: dict = {}
        closed    = set()
        explored  = []

        while open_heap:
            _f, g, r, c = heapq.heappop(open_heap)

            if (r, c) in closed:
                continue
            closed.add((r, c))
            explored.append((r, c))

            if (r, c) == (gr, gc):
                # Reconstruct path
                path, node = [], (gr, gc)
                while node in came_from:
                    path.append(node)
                    node = came_from[node]
                path.append((sr, sc))
                path.reverse()
                return path, explored

            for dr, dc, cost in MOVES:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                if grid[nr, nc] or (nr, nc) in closed:
                    continue
                tg = g + cost
                if tg < g_score.get((nr, nc), float('inf')):
                    g_score[(nr, nc)]   = tg
                    came_from[(nr, nc)] = (r, c)
                    heapq.heappush(open_heap, (tg + h(nr, nc), tg, nr, nc))

        return None, explored

    # =========================================================================
    # NAV2 FOLLOWPATH ACTION
    # =========================================================================

    def _follow_path(self, path: Path):
        """
        Send the A* path to Nav2's controller server via the FollowPath action.
        Nav2's own controller (DWB / RegulatedPurePursuit) will drive the robot,
        handling real-time obstacle avoidance along the path.
        """
        self.get_logger().info('Waiting for Nav2 FollowPath action server...')
        if not self.follow_path_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('FollowPath server not available — aborting.')
            return

        goal_msg = FollowPath.Goal()
        goal_msg.path = path
        # Use the default controller configured in Nav2 (e.g. FollowPath)
        goal_msg.controller_id = ''

        self.get_logger().info(
            f'Sending {len(path.poses)}-pose path to Nav2 FollowPath...'
        )

        future = self.follow_path_client.send_goal_async(
            goal_msg, feedback_callback=self._follow_feedback_cb
        )
        future.add_done_callback(self._follow_goal_response_cb)

    def _follow_goal_response_cb(self, future):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().error('FollowPath goal rejected by Nav2.')
            return
        self.get_logger().info('Nav2 accepted path — robot is navigating.')
        handle.get_result_async().add_done_callback(self._follow_result_cb)

    def _follow_result_cb(self, future):
        self.get_logger().info('✓ Navigation complete (Nav2 FollowPath finished).')

    def _follow_feedback_cb(self, feedback_msg):
        # FollowPath feedback includes distance_to_goal and speed
        fb = feedback_msg.feedback
        self.get_logger().info(
            f'  dist to goal: {fb.distance_to_goal:.2f} m  '
            f'speed: {fb.speed:.2f} m/s',
            throttle_duration_sec=2.0,
        )

    # =========================================================================
    # VISUALISATION
    # =========================================================================

    def _publish_explored(self, explored: list, info):
        """
        Publish ALL explored nodes as a single MarkerArray.

        Why single publish instead of batched sleeps:
        - time.sleep() in a background thread does not advance sim time, so
          every marker in a batch loop gets an identical timestamp — RViz may
          ignore or deduplicate them.
        - A single large publish with TRANSIENT_LOCAL QoS is received reliably
          by RViz regardless of when the display was added.
        - The blue→green colour gradient encodes expansion order so the search
          tree is still clearly readable.

        Stamp = Time() (zero) tells RViz "display always, ignore time filtering."
        """
        now = self.get_clock().now().to_msg()
        res = info.resolution
        n   = max(len(explored) - 1, 1)

        markers = MarkerArray()

        # 1. DELETEALL — wipe any previous search tree
        dm = Marker()
        dm.header.frame_id = 'map'
        dm.header.stamp    = now
        dm.ns              = 'astar_explored'
        dm.action          = Marker.DELETEALL
        markers.markers.append(dm)

        # 2. One CUBE marker per explored cell
        for idx, (r, c) in enumerate(explored):
            wx, wy = self._cell_to_world((r, c), info)
            t = idx / n          # 0.0 = first expanded, 1.0 = last expanded

            mk = Marker()
            mk.header.frame_id    = 'map'
            mk.header.stamp       = now   # ← zero stamp: sim-time safe
            mk.ns                 = 'astar_explored'
            mk.id                 = idx + 1      # +1 so id=0 is reserved for DELETEALL
            mk.type               = Marker.CUBE
            mk.action             = Marker.ADD
            mk.pose.position.x    = wx
            mk.pose.position.y    = wy
            mk.pose.position.z    = 0.005
            mk.pose.orientation.w = 1.0
            mk.scale.x = res * 0.85
            mk.scale.y = res * 0.85
            mk.scale.z = 0.01
            # Blue (first expanded) → Cyan → Green (last expanded)
            mk.color.r = 0.0
            mk.color.g = t
            mk.color.b = 1.0 - 0.6 * t
            mk.color.a = 0.6
            mk.lifetime.sec = 0   # 0 = persist forever (until DELETEALL)
            markers.markers.append(mk)

        self.explored_pub.publish(markers)
        self.get_logger().info(
            f'✓ Published {len(explored)} explored nodes to /astar/explored.'
        )

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _build_path_msg(
        self, path_world: list[tuple[float, float]], frame_id: str = 'map'
    ) -> Path:
        msg = Path()
        msg.header.frame_id = frame_id
        msg.header.stamp    = rclpy.time.Time().to_msg()   # zero stamp: sim-time safe
        for wx, wy in path_world:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x    = wx
            ps.pose.position.y    = wy
            ps.pose.position.z    = 0.0
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        return msg

    def _world_to_cell(self, world: tuple, info) -> tuple[int, int]:
        col = int((world[0] - info.origin.position.x) / info.resolution)
        row = int((world[1] - info.origin.position.y) / info.resolution)
        return (row, col)

    def _cell_to_world(self, cell: tuple, info) -> tuple[float, float]:
        r, c = cell
        wx = info.origin.position.x + (c + 0.5) * info.resolution
        wy = info.origin.position.y + (r + 0.5) * info.resolution
        return (wx, wy)

    def _stop_robot(self):
        stop = TwistStamped()
        stop.header.frame_id   = 'base_link'
        stop.header.stamp      = self.get_clock().now().to_msg()
        stop.twist.linear.x    = 0.0
        stop.twist.angular.z   = 0.0
        self.cmd_pub.publish(stop)


# =============================================================================
def main(args=None):
    rclpy.init(args=args)
    node = AStarNav()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._stop_robot()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()