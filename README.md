# Universal platform for sorting known and unknown objects by cells

![logo](docs/images/grasping_cell.svg)

[documentation in Russian](README_ru.md)

This repository contains the files of the **Laboratory of Biomechatronics and Energy Efficient Robotics of ITMO University** project to create a universal platform for sorting known and unknown objects by cells. The system detects / classifies objects in space according to technical vision data, sorts them according to the cells corresponding to each object, moving in space with obstacles. The system also has a function of automated retraining, which allows you to study new objects and sort them without reconfiguring the software platform.

## Hardware
The system is based on the manipulator [Kuka LBR iiwa](https://www.kuka.com/ru-ru/%D0%BF%D1%80%D0%BE%D0%B4%D1%83%D0%BA%D1%86%D0%B8%D1%8F-%D1%83%D1%81%D0%BB%D1%83%D0%B3%D0%B8/%D0%BF%D1%80%D0%BE%D0%BC%D1%8B%D1%88%D0%BB%D0%B5%D0%BD%D0%BD%D0%B0%D1%8F-%D1%80%D0%BE%D0%B1%D0%BE%D1%82%D0%BE%D1%82%D0%B5%D1%85%D0%BD%D0%B8%D0%BA%D0%B0/%D0%BF%D1%80%D0%BE%D0%BC%D1%8B%D1%88%D0%BB%D0%B5%D0%BD%D0%BD%D1%8B%D0%B5-%D1%80%D0%BE%D0%B1%D0%BE%D1%82%D1%8B/lbr-iiwa). This is a collaborative manipulator with 7 degrees of freedom, which is absolutely safe for a person and can work next to him without the risk of damage or damage.
The vision system is based on a camera [Intel Realsense D435i](https://www.intelrealsense.com/depth-camera-d435i/). A stereo and a depth cameras allow you to determine the shape, size of objects in space and their distances with great accuracy.

## Software
The software platform is based on a framework [MoveIt!](https://moveit.ros.org/) and consists of an object detection/classification module, a motion planning module, an object capture module, and an additional training module. Interaction between modules occurs through a finite state machine (FSM). Sorting mode work cycle: Go to start position -> Detection and classification of objects -> Detection of the nearest object and segmentation -> Generation of possible configurations of the manipulator for capturing objects -> Planning the movement of the robot from the current configuration to the configuration for capturing -> Moving to a new configuration - > Capturing the object -> Planning the movement to the cell -> Lowering the object into the cell -> Return to the starting position. The block diagram of the system operation is shown in the figure below:

![flowchart](docs/images/flow_chart.png)

Interaction with the system occurs through the user interface (GUI interface), which enables/disables the system, as well as switching between modes of automated sorting and additional training of new objects. The finite state machine (FSM) sends requests to the modules and, based on the responses, determines the next actions of the system. The general architecture of the system is shown in the figure below:

![architecture](docs/images/architecture.png)

Detailed description of modules:

1. [Detection and classification of objects](docs/cv.md)
2. [Pose estimation](docs/grasp.md)
3. [Motion planning](docs/plan.md)
4. [Work of the services and FSM](docs/fsm.md)

## How to use

### Prerequisites

- [Ubuntu 20.04,03 Focal Fossa LTS](https://releases.ubuntu.com/20.04/)
- [Robot operating system (ROS) Noetic Ninjemys](http://wiki.ros.org/noetic)
- [Intel RealSense SDK 2.0](https://www.intelrealsense.com/sdk-2/)
- [MoveIt!](https://moveit.ros.org/install/)

### Installation

1. Download the official repository from [Kuka LBR iiwa for MoveIt!](https://github.com/IFL-CAMP/iiwa_stack) and install it into your workspace according to the instructions in the repository.
2. 2. Download the official repository from [Intel Realsense D435i](https://github.com/IntelRealSense/realsense-ros) and install it into your workspace according to the instructions in the repository.
3. Download the files of this repository to the workspace and start building the project (catkin build or catkin_make).
4. (Optional) Set up a local network between the robot and the computer (not required to run the simulation on the computer).

5. (Optional) Install the scheduler library [OMPL with Modified Intelligent Bidirectional Fast Exploring Random Tree](https://github.com/IDovgopolik/ompl) (you can use the OMPL scheduler library built into MoveIt! to work).
6. Start the system:

```bash
roslaunch iiwa_moveit move_group.launch
roslaunch iiwa_move_group_interface move_group_interface_iiwa.launch
```
**Can be used!**