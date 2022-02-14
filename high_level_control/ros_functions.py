import numpy as np

import rospy
from std_srvs.srv import Trigger
from sensor_msgs.msg import Image, CameraInfo
import tf


from computer_vision.srv import SegmentAndClassifyService
from computer_vision.msg import SegmentAndClassifyResult
from contact_graspnet_planner.srv import ContactGraspNetPlanner
from utils import QuaternionToArr, get_center_coordinates, get_points_with_vectors, \
    get_points_with_vectors, get_poses_from_grasps, \
        get_rotation_by_vector, rosPose, PointToArr, get_list_of_poses, check_service
from high_level_control.srv import GoalPoses, GoalPosesRequest

import config as cfg

br = tf.TransformBroadcaster()
np.set_printoptions(precision=2)
# listener = tf.TransformListener()


def check_all_nodes():

    rospy.logwarn('Checking nodes:')
    service_list = ['/iiwa/check_state_validity', 
    '/cv_inference_service',
                    '/cv_train_service', '/GoalPoses', '/grasp_planner', '/gripper_state']
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
    topic_list = ['/camera/color/image_raw', '/camera/aligned_depth_to_color/image_raw', '/camera/color/camera_info', '/cv_results']
    type_list = [Image, Image, CameraInfo, SegmentAndClassifyResult]

    for topic, t in zip(topic_list,type_list):
        try:
            rospy.wait_for_message(topic, t, timeout=cfg.timeout)
            rospy.logwarn(f'{topic} [OK]')
            
        except rospy.ROSException as e:
            rospy.logerr(f'{topic} [NOT OK]')
            success = False

    return success


@check_service(fail_return_value=None)
def run_object_recognition():
    # check if service is available. If Not, use topic 
    try:
        srv_proxy = rospy.ServiceProxy(
            'cv_inference_service', SegmentAndClassifyService)
        response = srv_proxy()
        return response.results
        # return (response.results.masked_depth, response.results.mask, response.results.class_name, response.results.class_conf, response.results.class_dist)         
    except rospy.ServiceException as e:
        rospy.logwarn("CV service is not available, trying to get result from topic")
        
        results = rospy.wait_for_message('/cv_results', SegmentAndClassifyResult, timeout=5)
        return results
                 


@check_service(fail_return_value=None)
def moveObjectToBox(grasping_poses, class_index):
    
    try:
        moveToPoint = rospy.ServiceProxy(
            '/GoalPoses', GoalPoses)
        request = GoalPosesRequest()
        request.GoalBoxIndex = int(class_index)
        request.WorkingMode = 0
        request.Poses = [a.pose for a in grasping_poses.grasps]
        request.Poses = get_poses_from_grasps(grasping_poses)

        print(f'Going to box: {request.GoalBoxIndex}')

        response = moveToPoint(request)

    except rospy.ServiceException as e:
        rospy.logerr(e)
        return 'Planning failed'

    if response.AllMotionWorked:
        return 'Moving sucessful'
    elif not response.GraspWorked:
        return 'Planning failed'


def checkPosition(goal_position, goal_orientation, cur_pose=None, target_frame=None):

    # Check if goal pose is the same as the current pose. Requires current pose or target_frame name (to obtain current pose from ros transforms)
    if cur_pose is target_frame is None:
        raise f"Can't get current pose"

    if cur_pose is None:
        listener = tf.TransformListener()
        listener.waitForTransform('world', target_frame, rospy.Time(0), rospy.Duration(secs=5))
        (position,rot) = listener.lookupTransform('world', target_frame, rospy.Time(0))

        cur_position = np.array(position)
        cur_orientation = np.array(rot)
    else:
        cur_position = PointToArr(cur_pose.position)
        cur_orientation = QuaternionToArr(cur_pose.orientation)

    cur_orientation = np.absolute(cur_orientation) # workaround for handling negative quaternion near limits

    positions_are_same = np.allclose(cur_position, goal_position, atol=1e-2)
    orientations_are_same = np.allclose(cur_orientation, goal_orientation, atol=1e-2) or np.allclose(-cur_orientation, goal_orientation, atol=1e-2)

    return positions_are_same and orientations_are_same

@check_service(fail_return_value=None)
def moveManipulatorToHome():
    # check if end-effector is already at initial position and move to this pose 
    pose_achieved = checkPosition(np.array(cfg.initial_position), np.array(cfg.initial_orientation), target_frame='iiwa_link_ee_grasp')

    if pose_achieved:
        rospy.loginfo('Already at initial pose!')
        return True
    else:
        try:
            # rospy.loginfo(f'current position: {trans}, current orientation quaternion: {rot}')
            # rospy.loginfo(f'current position: {np.array(cfg.initial_position)}, Init orientation quaternion: {np.array(cfg.initial_orientation)}')

            moveToPoint = rospy.ServiceProxy(
                '/GoalPoses', GoalPoses)
            request = GoalPosesRequest()
            request.WorkingMode = 3
            request.Poses = [rosPose(cfg.initial_position, cfg.initial_orientation)]
            response = moveToPoint(request)
            return True

        except rospy.ServiceException as e:
            rospy.logerr(e)
            return False


def generate_grasps(mask):
    # combine mask of desired object obtained from CV module and raw realsense RGB and D images
    color_im = rospy.wait_for_message('/camera/color/image_raw', Image)
    depth_im = rospy.wait_for_message(
        '/camera/aligned_depth_to_color/image_raw', Image)
    camera_info_ros = rospy.wait_for_message(
            '/camera/aligned_depth_to_color/camera_info', CameraInfo)

    grasp_planner = rospy.ServiceProxy(
        'grasp_planner', ContactGraspNetPlanner)


    response = grasp_planner(
        color_im,
        depth_im,
        camera_info_ros,
        mask
    )
    rospy.loginfo("Get {} grasps from the server.".format(
        len(response.grasps)))
    
    return response

@check_service(fail_return_value='Module not available')
def changePosition(depth_masked, attempt):
    # in case of failure change position few times according to found objects
    if attempt == 0:
        changePosition.poses = get_list_of_poses(PointToArr(get_center_coordinates(depth_masked)), shift=cfg.change_position_shift)

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
        
@check_service(fail_return_value=None)
def save_frame():

    # call service for saving the nearest found object
    try:
        srv_proxy = rospy.ServiceProxy(
            'cv_train_service', Trigger)
        response = srv_proxy()

        return response
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")
        return None

def learn_object(mask):
    # 1. get grasp poses
    grasping_poses = generate_grasps(mask)
    # request.Poses = [a.pose for a in grasping_poses.grasps]
    if not grasping_poses:
        return 'Grasps not generated'
    # 2. Move object to a plane
    try:
        moveToPoint = rospy.ServiceProxy(
            '/GoalPoses', GoalPoses)
        request = GoalPosesRequest()
        request.WorkingMode = 1

        request.Poses = get_poses_from_grasps(grasping_poses)
        response = moveToPoint(request)
    except rospy.ServiceException as e:
        rospy.logerr(e)
    # 3. Generate N positions
    points, vectors = get_points_with_vectors(cfg.plane_obj_coordinates, cfg.learn_n_points)

    # 4. Move and save frames

    saved_frames = 0

    points_to_keep_idxs = [0, 3, 4] # workaround

    for idx, (point, vector) in enumerate(zip(points[points_to_keep_idxs], vectors[points_to_keep_idxs])):
        rospy.loginfo(f'moving to point {idx+1}/{len(points_to_keep_idxs)}')
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
                save_frame()
                saved_frames += 1
            # return 'Object saved'
        except rospy.ServiceException as e:
            rospy.logerr(e)
            # return 'Object not saved'

    end_train = rospy.ServiceProxy(
                '/cv_end_train_service', Trigger)
    is_saved = end_train()
    
    
    rospy.loginfo(f'Saved {saved_frames}/{len(points_to_keep_idxs)} frames')
    rospy.loginfo('Going to grasp object')
    # 4. go above object to detect it

    request = GoalPosesRequest()


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

    if not is_saved:
        return 'Object not saved'

    return 'Object saved' if saved_frames / cfg.learn_n_points  > 0.5 else 'Object not saved'


if __name__ == '__main__':

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
