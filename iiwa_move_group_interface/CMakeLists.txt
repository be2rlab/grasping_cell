cmake_minimum_required(VERSION 3.1.3)
project(iiwa_move_group_interface)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(catkin REQUIRED
  COMPONENTS
    interactive_markers
    moveit_core
    moveit_visual_tools
    moveit_ros_planning
    moveit_ros_planning_interface
    moveit_ros_perception
    pluginlib
    geometric_shapes
    pcl_ros
    pcl_conversions
    rosbag
    tf2_ros
    tf2_eigen
    tf2_geometry_msgs
    roscpp
    rospy
    std_msgs
    message_generation
    geometry_msgs
    uhvat_ros_driver
)

find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED system filesystem date_time thread)

set(THIS_PACKAGE_INCLUDE_DIRS
  doc/interactivity/include
)

add_service_files(
   FILES
   GoalPoses.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

catkin_package(
  LIBRARIES
  INCLUDE_DIRS
  CATKIN_DEPENDS
    moveit_core
    moveit_visual_tools
    moveit_ros_planning_interface
    interactive_markers
    tf2_geometry_msgs
    uhvat_ros_driver
  DEPENDS
    EIGEN3
)

###########
## Build ##
###########

include_directories(${THIS_PACKAGE_INCLUDE_DIRS}
                    ${catkin_INCLUDE_DIRS}
                    ${Boost_INCLUDE_DIR}
                    ${EIGEN3_INCLUDE_DIRS})

add_subdirectory(doc/move_group_interface)


