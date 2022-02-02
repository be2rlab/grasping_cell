# Work of the services and FSM

## General description

The scene consists of a manipulator with a gripper and a camera located on it, a box with unsorted objects, several boxes intended for objects of a certain class, a wall with two windows and a table on which everything is located. Inside the box lie various classes. The initial position of the manipulator is located above the box with unsorted items with the working body lowered down. A wall with holes is an obstacle to the movement of the manipulator. Scene in the MoveIt! shown in the figure: 

![scene](images/scene.png)

Items in the box are recognized by the computer vision system, configurations for capture are built for the nearest object, and the item is transferred to the box corresponding to the recognized class. If an object is found, but the computer vision model is not sure about the prediction, then a small change in perspective occurs with an offset of a few centimeters. With repeated failed recognition attempts, the process of retraining to a new object is started.
There are three stages in the learning process:
1. Moving an object on the plane of the table to a given point;
2. Collection of images of the object from different angles;
3. Move the object to the box.
A spiral trajectory was built to collect data. On the trajectory, N points are selected evenly along the vertical axis and their orientation is calculated so that the Z axis of the camera is directed to the object in the center of the spiral. We also fix the X-axis of the camera so that it lies in the horizontal plane to limit the rotation. The trajectory is shown in the figure: 

![traj](images/traj.png)

The coordinates of all the boxes on the scene are predefined and it is assumed that the scene is static. 

## Finite state machine (FSM)

To implement control, a state machine based on the [Smach](http://wiki.ros.org/smach) was implemented. It allows you to build complex, hierarchical state machines and keep track of the state. There is also integration with ROS.
The system has 3 main modules:
1. Computer vision with object recognition;
2. Generation of positions for capture;
3. Planning and implementation of the movement.
There was also an Intel RealSense camera module that published data to topics. It was decided to communicate with the modules through services.
The computer vision module in inference mode, upon receiving a request from the client, receives RGB and Depth images from camera topics, performs segmentation, classification, and sends a response containing the class of the nearest object, model confidence, object mask, and distance to the nearest point in the feature space. Also, the original image with the contours of recognized objects is published in the topic in order to be able to monitor the operation of the module. For the retraining mode, another service is provided, implemented through a ROS trigger, which saves the object closest to the camera into memory.
The structure of the message in the response is shown below: 

```txt
std_msgs/Header header
sensor_msgs/Image mask
sensor_msgs/Image masked_depth
string class_name
float32 class_dist
float32 class_conf
```

Capturing position generation module, having received the object mask, receives the depth image and internal parameters of the camera from the camera topics. On the basis of these data, a point cloud of the visible part of the scene is built and, using the ContactGraspNet algorithm, positions are generated to capture the object. In response, the service sends an array of positions with a numerical assessment of the model's confidence in these configurations. The structure of the service is shown below: 

```txt
sensor_msgs/Image mask
---
# response params
bool success
ContactGrasp[] grasps
```

The movement planning and implementation module has several fields: an array of positions Poses, the number of the operating mode WorkingMode and the index of the final box GoalBoxIndex. There are several modes of operation:

0. Grasp an object, move it from the current position to the given box and return to the starting position;
1. Grasp an object and move it to a fixed place on the plane for further retraining;
2. Move the camera to a position from the Poses field;
3. Return to the starting position.

As a response, the service sends the AllMotionWorked field with a True/False value. If there are no toolpaths, the generated grab positions will return False in the GraspWorked field. The message structure for the ROS service is shown below: 

```txt
geometry_msgs/Pose[] Poses
int32 GoalBoxIndex # Important only in WorkingMode=0
int32 WorkingMode
# Working modes:
# 0: Grasp, move from start box to goal box and return to init state(length of Poses > 0)
# 1: Grasp and move object to a fixed point on a plane to learn (lenth of poses > 0); 
# 2: Change position to point specified in "Poses" field (length of Poses = 1); used in learning process
# 3: return to initial state (Poses = [])
---
bool GraspWorked
bool AllMotionWorked
```

To prevent crashes in the control program, each service call was placed inside a Try-except block. The structure is shown below: 

```txt
def save_feature():
    try:
        srv_proxy = rospy.ServiceProxy(
            'segmentation_train_service', Trigger)
        response = srv_proxy()
        return response
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")
        return None
```

Also, to monitor the status of services and topics, a function was written that checks all the necessary nodes, the code of which is presented below: 

```txt
def check_all_nodes():
    rospy.logwarn('Checking nodes:')
    service_list = ['segmentation_inference_service',
                    'segmentation_train_service', 'GoalPoses', 'response', 'gripper_state']
    success = True
    for service in service_list:
        try:
            rospy.wait_for_service(service, timeout=cfg.timeout)
            rospy.logwarn(f'{service} [OK]')
        except rospy.ROSException as e:
            rospy.logerr(f'{service} [NOT OK]')
            success = False
    rospy.logwarn("")
    rospy.logwarn("Checking topics:")
    topic_list = ['camera/color/image_raw', 'camera/aligned_depth_to_color/image_raw', 'camera/color/camera_info']
    type_list = [Image, Image, CameraInfo]
    for topic, t in zip(topic_list,type_list):
        try:
            rospy.wait_for_message(topic, t, timeout=cfg.timeout)
            rospy.logwarn(f'{topic} [OK]')
        except rospy.ROSException as e:
            rospy.logerr(f'{topic} [NOT OK]')
            success = False
    if not success:
        success = False
    return True
```

## Estimation of orientation

For the starting point and points above each of the boxes, rotations are predefined so that the working element is located either horizontally or vertically. For points when changing the angle and during additional training, it is necessary to calculate the rotation. It is most convenient to set the rotation of coordinate systems, knowing the final location of the basis vectors, through the rotation matrix.
For the case of changing the angle with unrecognized objects, several shifts are initially set in the camera's coordinate system. In this case, 6 points were set: with an offset of 5 cm along each axis. Then we translate all the points into the base coordinate system. Next, you need to get the vectors corresponding to the direction of the Z vector for each point in the direction of the segmented object. To do this, the coordinates of the point are subtracted from the coordinates of the center of the object. Next, we restrict the Y vector by aligning the first two components with the components of the corresponding vector of the camera's coordinate system. Thus, we can uniquely specify the third component of the Y vector, based on the requirement that the angle between the Y and Z vectors is 90 degrees and their dot product is zero. 

## Safety

Collaborative robots are absolutely safe for humans. But the system provides additional protection. The speed is set programmatically and can reach unsafe values. Human reaction time is on average 100 milliseconds. To ensure safety in the controller, the speed is limited so that a person always has time to react. If the program speed has a higher value, then if the limit is exceeded during movement, the robot will interrupt the execution and block the movement until the speed in the program is reduced. 