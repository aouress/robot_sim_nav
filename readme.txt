Instructions for reproducing results can be found below:

Link to unlisted YouTube video for video demo: https://youtu.be/f0p6KRRATeE

For the final project demo, I was instructed / recommended by Dr. Islam to write a custom path planner using A* or similar and display results of the algorithm's search to rviz. 

This Robot Navigation project completes the following goals: 
- In an unknown world, using SLAM and an exploration package, generate a fixed map of the world 
- Using the fixed map, switch to entirely localization (using AMCL) to allow for a more accurate represenation of robot state. 
- Generate paths to selected goals using A* and send these goals to appropriate Nav2 navigation methods 
- The path and explore nodes are then displayed in Rviz with a color gradient to show recency of the explored nodes (green = more recently explored). 

- for all files, make sure the path to the project root (where you colcon build) is ~/code/robot_sim_nav/

Running SLAM and Explore 
In the terminal run 
>> export TURTLEBOT3_MODEL=burger
>> cd ~/code/robot_sim_nav/ 
>> source install/setup.bash
>> ros2 run mission_manager slam_and_explore

Running Custom Path Planner for Navigation with Relocalization
In each of the following terminals, run
>> export TURTLEBOT3_MODEL=burger
- Terminal 1
>> ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
- Terminal 2
>> ros2 launch turtlebot3_navigation2 navigation2.launch.py use_sim_time:=True map:=$HOME/code/robot_sim_nav/src/maps/map.yaml
- Terminal 3
>> cd ~code/robot_sim_nav
>> source install/setup.bash
>> ros2 run path_to_nav astar_nav
- To visualize the path planning and exploration of the grid, you need to add the views in Rviz (see video for exact steps).
