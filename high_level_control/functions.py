from time import time
from pandas import Timedelta, TimedeltaIndex
import rospy
from std_srvs.srv import Trigger, TriggerResponse
from geometry_msgs.msg import Pose, Point, Quaternion
from sensor_msgs.msg import Image, CameraInfo
# import open3d as o3d
# from cv_bridge import CvBridge
# import cv2 as cv
import numpy as np
# from matplotlib import pyplot as plt
import tf

from segmentation.srv import SegmentAndClassifyService, SegmentAndClassifyServiceRequest, SegmentAndClassifyServiceResponse
from segmentation.msg import SegmentAndClassifyResult
from contact_graspnet_planner.srv import ContactGraspNetAnswer
from utils import ArrayToRos, QuaternionToArr, get_center_coordinates, get_points_with_vectors, get_points_with_vectors, get_poses_from_grasps, get_proj_point, get_rotation_by_vector, rosPose, PointToArr, get_list_of_poses
from kuka_smach_control.srv import GoalPoses, GoalPosesRequest, GoalPosesResponse



import config as cfg

br = tf.TransformBroadcaster()
np.set_printoptions(precision=2)

def check_all_nodes():

    rospy.logwarn('Checking nodes:')
    service_list = ['/iiwa/check_state_validity', 
    #'/segmentation_inference_service',
                    '/segmentation_train_service', '/GoalPoses', '/response', '/gripper_state']
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
    topic_list = ['/camera/color/image_raw', '/camera/aligned_depth_to_color/image_raw', '/camera/color/camera_info', '/segm_results']
    type_list = [Image, Image, CameraInfo, SegmentAndClassifyResult]

    for topic, t in zip(topic_list,type_list):
        try:
            rospy.wait_for_message(topic, t, timeout=cfg.timeout)
            rospy.logwarn(f'{topic} [OK]')
            
        except rospy.ROSException as e:
            rospy.logerr(f'{topic} [NOT OK]')
            success = False
    # if not success:
    #     success = False
    return True



def run_segmentation():
    service_is_available = True
    try:
        rospy.wait_for_service('segmentation_inference_service', timeout=cfg.timeout)
    except rospy.ROSException as e:
        service_is_available = False
        rospy.logwarn('segmentation_inference_service not available, taking results from topic')

    try:
        if service_is_available:
            srv_proxy = rospy.ServiceProxy(
                'segmentation_inference_service', SegmentAndClassifyService)
            response = srv_proxy()

            return (response.results.masked_depth, response.results.mask, response.results.class_name, response.results.class_conf, response.results.class_dist)
        else:
            results = rospy.wait_for_message('/segm_results', SegmentAndClassifyResult)
            return (results.masked_depth, results.mask, results.class_name, results.class_conf, results.class_dist)
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")
        return None


def move(userdata):
    grasping_poses = userdata.grasping_poses
    class_index = userdata.class_name
    try:
        moveToPoint = rospy.ServiceProxy(
            '/GoalPoses', GoalPoses)
        request = GoalPosesRequest()
        request.GoalBoxIndex = int(class_index)
        request.WorkingMode = 0
        request.Poses = [a.pose for a in grasping_poses.grasps]
        request.Poses = get_poses_from_grasps(grasping_poses)

        print(f'Going to box: {request.GoalBoxIndex}')
        print(request.Poses[0].position)

        # rospy.logwarn(len)
        response = moveToPoint(request)

    except rospy.ServiceException as e:
        rospy.logerr(e)
        return 'Planning failed'

    if response.AllMotionWorked:
        return 'Moving sucessful'
    elif not response.GraspWorked:
        return 'Planning failed'


def moveManipulatorToHome():
    # check if end-effector is already at initial position and move to this pose 
    listener = tf.TransformListener()
    listener.waitForTransform('world', 'iiwa_link_ee_grasp', rospy.Time(0), rospy.Duration(secs=5))
    (trans,rot) = listener.lookupTransform('world', 'iiwa_link_ee_grasp', rospy.Time(0))

    trans = np.array(trans)
    rot = np.array(rot)

    rot = np.absolute(rot) # handle 

    translation_is_init = np.allclose(trans, np.array(cfg.initial_position), atol=1e-2)
    rotation_is_init = np.allclose(rot, np.array(cfg.initial_orientation), atol=1e-2) or np.allclose(-rot, np.array(cfg.initial_orientation), atol=1e-2)

    if translation_is_init and rotation_is_init:
        rospy.loginfo('Already at initial pose!')
        return True
    else:
        try:
            rospy.loginfo(f'current position: {trans}, current orientation quaternion: {rot}')
            rospy.loginfo(f'current position: {np.array(cfg.initial_position)}, Init orientation quaternion: {np.array(cfg.initial_orientation)}')

            moveToPoint = rospy.ServiceProxy(
                '/GoalPoses', GoalPoses)
            request = GoalPosesRequest()
            request.WorkingMode = 3
            request.Poses = [rosPose(cfg.initial_position, cfg.initial_orientation)]
            response = moveToPoint(request)
            (trans,rot) = listener.lookupTransform('world', 'iiwa_link_ee_grasp', rospy.Time(0))
            # rospy.logerr((trans, rot))
            return True

        except rospy.ServiceException as e:
            rospy.logerr(e)
            return False


def generate_grasps(mask):

    rospy.wait_for_service('/response')
    try:
        srv_proxy = rospy.ServiceProxy(
            '/response', ContactGraspNetAnswer)
        response = srv_proxy(mask)

        rospy.logwarn(f'Generated {len(response.grasps)} grasps')

        return response if len(response.grasps) != 0 else None
    except rospy.ServiceException as e:
        rospy.logerr(e)
        return None


def changePosition(depth_masked, attempt):

    if attempt == 0:
        changePosition.poses = get_list_of_poses(PointToArr(get_center_coordinates(depth_masked)), shift=cfg.change_position_shift)

    # point = get_center_coordinates(depth_masked)
    pose = rosPose(*changePosition.poses[attempt])

    try:
        moveToPoint = rospy.ServiceProxy(
            '/GoalPoses', GoalPoses)
        request = GoalPosesRequest()
        request.WorkingMode = 2
        request.Poses = [pose]
        br.sendTransform(*changePosition.poses[attempt],
                        rospy.Time.now(),
                        'goal_pose',
                        "world")
        print(pose.position)
        response = moveToPoint(request)
        print(response)
        return 'Position changed' if response.AllMotionWorked else 'Moving failed'
    except rospy.ServiceException as e:
        rospy.logerr(e)
        return 'Moving failed'
        

def save_feature():
    try:
        srv_proxy = rospy.ServiceProxy(
            'segmentation_train_service', Trigger)
        response = srv_proxy()

        return response
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")
        return None

def save_object(depth_masked):
    # 1. get grasp poses

    grasping_poses = generate_grasps(depth_masked)
    # request.Poses = [a.pose for a in grasping_poses.grasps]
    if not grasping_poses:
        return 'Grasps not generated'
    # 1. Move object to a plane
    try:
        moveToPoint = rospy.ServiceProxy(
            '/GoalPoses', GoalPoses)
        request = GoalPosesRequest()
        request.WorkingMode = 1

        request.Poses = get_poses_from_grasps(grasping_poses)
        response = moveToPoint(request)
    except rospy.ServiceException as e:
        rospy.logerr(e)
    # 2. Generate N positions
    points, vectors = get_points_with_vectors(cfg.plane_obj_coordinates, cfg.learn_n_points)

    # 3. Move and save frames

    saved_frames = 0

    for idx, (point, vector) in enumerate(zip(points[[0, 3, 4]], vectors[[0, 3, 4]])):
        rospy.loginfo(f'moving to point {idx+1}/3')
        print(point)
        request = GoalPosesRequest()

        quat_array = get_rotation_by_vector(vector)

        br.sendTransform(point,
                        quat_array,
                        rospy.Time.now(),
                        'goal_pose',
                        "world")

        pose = rosPose(point, quat_array)

        request.Poses = [pose]
        request.WorkingMode = 2

        try:
            moveToPoint = rospy.ServiceProxy(
                '/GoalPoses', GoalPoses)
            response = moveToPoint(request)
            print(response)
            rospy.sleep(2)
            if response.AllMotionWorked:
                rospy.loginfo(f'saving features')
                save_feature()
                saved_frames += 1
            # return 'Object saved'
        except rospy.ServiceException as e:
            rospy.logerr(e)
            # return 'Object not saved'

    end_train = rospy.ServiceProxy(
                '/segmentation_end_train_service', Trigger)
    end_train()
    
    rospy.loginfo(f'Saved {saved_frames}/3 frames')
    rospy.loginfo('Going to grasp object')
    # 4. got above object to detect it

    request = GoalPosesRequest()
    # br.sendTransform(point,
    #                 quat_array,
    #                 rospy.Time.now(),
    #                 'goal_pose',
    #                 "world")

    point = cfg.plane_obj_coordinates.copy()
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    point[0] = 100 
    # !!!!!!!!!!!!!!!!!Big workaround

    point[2] += 0.3

    pose = rosPose(point, cfg.plane_obj_orientation)

    request.Poses = [pose]
    request.WorkingMode = 2

    try:
        moveToPoint = rospy.ServiceProxy(
            '/GoalPoses', GoalPoses)
        response = moveToPoint(request)
        print(response)
        rospy.loginfo(f'saving features')
        # save_feature()
        saved_frames += 1
    except rospy.ServiceException as e:
        rospy.logerr(e)

    return 'Object saved' if saved_frames / cfg.learn_n_points  > 0.5 else 'Object not saved'


if __name__ == '__main__':
    # response = run_segmentation_srv()

    # generate_grasps(1)
    rospy.init_node("functions", log_level=rospy.INFO)
    # save_object()
    obj_camera_position = np.array([0.0, 0.0, 0.3])
    # obj_camera_position = np.array([-0.03243015286243519, 0.5725615182684507, 0.5093513058937228])
    poses = get_list_of_poses(obj_camera_position, shift=0.1, camera_frame='iiwa_link_ee_camera', base_frame='world')

    br = tf.TransformBroadcaster()

    
    rate = rospy.Rate(10.0)

    while not rospy.is_shutdown():
        br.sendTransform(obj_camera_position,
                             [0, 0, 0, 1],
                             rospy.Time.now(),
                             'object',
                             "iiwa_link_ee_camera")
        for idx, (point, quat) in enumerate(poses):

            print(point, quat)

            br.sendTransform(point,
                             quat,
                             rospy.Time.now(),
                             f'{idx}',
                             "world")
            rate.sleep()



    # print(response[0])
