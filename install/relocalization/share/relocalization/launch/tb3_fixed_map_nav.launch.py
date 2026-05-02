#
# Example:
# ros2 launch your_package localization_nav_pipeline.launch.py \
#   world:=turtlebot3_house.world \
#   robot:=waffle \
#   map:=$HOME/code/robot_sim_nav/src/maps/map.yaml

import os

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution

from launch.conditions import LaunchConfigurationEquals
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():

    use_sim_time = LaunchConfiguration("use_sim_time")
    map_file = LaunchConfiguration("map")
    world = LaunchConfiguration("world")
    robot = LaunchConfiguration("robot")

    scenario = LaunchConfiguration("scenario")

    default_map = os.path.join(
        os.environ["HOME"],
        "code/robot_sim_nav/src/maps/map.yaml"
    )

    return LaunchDescription([

        # -------------------------
        # Launch Arguments
        # -------------------------
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="True"
        ),

        DeclareLaunchArgument(
            "map",
            default_value=default_map
        ),

        # Choose between available worlds for tb3
        DeclareLaunchArgument(
            "scenario",
            default_value="sandbox"
        ),

        # burger / waffle / waffle_pi
        DeclareLaunchArgument(
            "robot",
            default_value="burger"
        ),

        # Set TurtleBot3 model for spawned robot
        SetEnvironmentVariable(
            name="TURTLEBOT3_MODEL",
            value=robot
        ),

        # -------------------------
        # Gazebo + Robot Spawn
        # -------------------------
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare("turtlebot3_gazebo"),
                    "launch",
                    "turtlebot3_world.launch.py"
                ])
            ),
            condition=LaunchConfigurationEquals(
                "scenario",
                "sandbox"
            ),
            launch_arguments={
                "use_sim_time": use_sim_time,
            }.items()
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare("turtlebot3_gazebo"),
                    "launch",
                    "turtlebot3_house.launch.py"
                ])
            ),
            condition=LaunchConfigurationEquals(
                "scenario",
                "house"
            ),
            launch_arguments={
                "use_sim_time": use_sim_time,
            }.items()
        ),

        # -------------------------
        # Nav2 Bringup
        # -------------------------
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare("turtlebot3_navigation2"),
                    "launch",
                    "navigation2.launch.py"
                ])
            ),
            launch_arguments={
                "use_sim_time": use_sim_time,
                "map": map_file
            }.items()
        ),

        # -------------------------
        # Custom Relocalization Node
        # -------------------------
        Node(
            package="relocalization",
            executable="relocalize_then_nav",
            name="relocalize_then_nav",
            output="screen",
            parameters=[{
                "use_sim_time": use_sim_time
            }]
        ),
    ])