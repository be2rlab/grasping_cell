import rospy
from std_srvs.srv import Trigger, TriggerResponse
from geometry_msgs.msg import Pose, Point, Quaternion
# import open3d as o3d
# from sensor_msgs.msg import Image, CameraInfo
# from cv_bridge import CvBridge
# import cv2 as cv
import numpy as np
# from matplotlib import pyplot as plt
import tf

from segmentation.srv import SegmentAndClassifyService, SegmentAndClassifyServiceRequest, SegmentAndClassifyServiceResponse
from segmentation.msg import SegmentAndClassifyResult
from contact_graspnet_planner.srv import ContactGraspNetAnswer
from utils import get_center_coordinates, get_points_with_vectors, get_points_with_vectors, get_proj_point, get_rotation_by_vector
from kuka_smach_control.srv import GoalPoses, GoalPosesRequest, GoalPosesResponse

import config as cfg


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


    topic_list = ['camera/color/image_raw', 'camera/aligned_depth_to_color/image_raw', 'camera/color/camera_info']

    all_topics = rospy.get_published_topics()
    # for topic in topic_list:

    #     try:
    #         rospy.wait_for_message(topic, None, timeout=cfg.timeout)
    #         rospy.logwarn(f'{topic} [OK]')
    #     except rospy.ROSException as e:
    #         rospy.logerr(f'{topic} [NOT OK]')
    return success



def run_segmentation():

    service_is_available = True
    try:
        rospy.wait_for_service('segmentation_inference_service', timeout=cfg.timeout)
    except rospy.ROSException as e:
        service_is_available = False
        rospy.logerr('segmentation_inference_service not available, taking results from topic')


    try:
        if service_is_available:
            srv_proxy = rospy.ServiceProxy(
                'segmentation_inference_service', SegmentAndClassifyService)
            response = srv_proxy()

            return (response.results.masked_depth, response.results.class_name, response.results.class_conf, response.results.class_dist)
        else:
            results = rospy.wait_for_message('/segm_results', SegmentAndClassifyResult)
            return (results.masked_depth, results.class_conf, results.class_dist)
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")
        return None


def move(grasping_poses):

    
    try:
        moveToPoint = rospy.ServiceProxy(
            '/GoalPoses', GoalPoses)
        request = GoalPosesRequest()
        request.GoalBoxIndex = 2
        request.WorkingMode = 2
        request.Poses = [a.pose for a in grasping_poses.grasps]

        for i in range(len(request.Poses)):
            # request.Poses[i].position = get_proj_point(
            #     grasping_poses.grasps[i].pose, grasping_poses.grasps[i].contact_point)
            request.Poses[i].position = grasping_poses.grasps[i].contact_point
        request.Poses = request.Poses[-cfg.plan_n_grasps:][::-1]
        request.Poses = [request.Poses[0]]

        print(request.Poses[0].position)

        # rospy.logwarn(len)
        response = moveToPoint(request)
        exit()

    except rospy.ServiceException as e:
        rospy.logerr(e)
        exit()
        return 'Planning failed'

    if not response.GraspWorked:
        return 'Planning failed'
    elif response.AllMotionWorked:
        return 'Moving sucessful'


def moveManipulatorToHome():
    try:
        moveToPoint = rospy.ServiceProxy(
            '/GoalPoses', GoalPoses)
        request = GoalPosesRequest()
        request.WorkingMode = 4
        request.Poses = []
        response = moveToPoint(request)
        return True

    except rospy.ServiceException as e:
        rospy.logerr(e)
        return False


def generate_grasps(depth_masked):
    rospy.wait_for_service('/response')

    try:
        srv_proxy = rospy.ServiceProxy(
            '/response', ContactGraspNetAnswer)
        response = srv_proxy(depth_masked)

        rospy.logwarn(f'Generated {len(response.grasps)} grasps')

        return response
    except rospy.ServiceException as e:
        rospy.logerr(e)
        return None


def changePosition(depth_masked):

    pose = get_center_coordinates(depth_masked)

    try:
        moveToPoint = rospy.ServiceProxy(
            '/GoalPoses', GoalPoses)
        request = GoalPosesRequest()
        request.WorkingMode = 2
        request.Poses = [pose]
        response = moveToPoint(request)
        print(response)
    except rospy.ServiceException as e:
        rospy.logerr(e)


def save_feature():

    try:
        srv_proxy = rospy.ServiceProxy(
            'segmentation_train_service', Trigger)
        response = srv_proxy()

        print(response)

        return response
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")
        return None



def save_object(depth_masked):
    br = tf.TransformBroadcaster()
    # 1. get grasp poses

    # resp = generate_grasps(depth_masked)


    # 1. Move object to a plane
    # try:
    #     moveToPoint = rospy.ServiceProxy(
    #         '/GoalPoses', GoalPoses)
    #     request = GoalPosesRequest()
    #     request.WorkingMode = 1
    #     request.Poses = grasping_pose
    #     response = moveToPoint(request)
    # except rospy.ServiceException as e:
    #     rospy.logerr(e)
    # # 2. Generate N positions
    points, vectors = get_points_with_vectors(cfg.plane_obj_coordinates, cfg.learn_n_points)
    # 2.5 check if all points are achievable ?

    # 3. Move and save frames

    for idx, (point, vector) in enumerate(zip(points, vectors)):
        rospy.loginfo(f'moving to point {idx+1}/{len(points)}')
        request = GoalPosesRequest()

        quat_array = get_rotation_by_vector(vector)
        quat = dict(zip(['x', 'y', 'z', 'w'], quat_array))

        br.sendTransform(point,
                            quat_array,
                            rospy.Time.now(),
                            'goal_pose',
                            "world")

        orientation = Quaternion(**quat)

        p = dict(zip(['x', 'y', 'z'], point))

        p = Pose(position=Point(**p), orientation=orientation)
        print(p)

        request.Poses = [p]
        request.WorkingMode = 3
        # request.GoalBoxIndex = 0
        # request.ChangeView = True

        try:
            moveToPoint = rospy.ServiceProxy(
                '/GoalPoses', GoalPoses)
            response = moveToPoint(request)
            print(response)
        except rospy.ServiceException as e:
            rospy.logerr(e)

        rospy.loginfo(f'saving features')
        save_feature()


if __name__ == '__main__':
    # response = run_segmentation_srv()

    # generate_grasps(1)
    rospy.init_node("functions", log_level=rospy.INFO)
    save_object()
    # print(response[0])
