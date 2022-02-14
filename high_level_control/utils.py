import math
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import Pose, Point, Quaternion
from std_msgs.msg import Header
import cv2 as cv
import matplotlib.pyplot as plt

from pyquaternion import Quaternion as pyQuat
import config as cfg

import rospy
import tf

from tf.transformations import quaternion_from_matrix, quaternion_matrix, euler_from_quaternion, quaternion_from_euler

br = tf.TransformBroadcaster()

bridge = CvBridge()

ArrayToRos = lambda rostype, array: rostype(**dict(zip(rostype.__slots__, array)))
rosPose = lambda position_array, orientation_arr: Pose(position=ArrayToRos(Point, position_array), orientation=ArrayToRos(Quaternion, orientation_arr))
PointToArr = lambda p: np.array([p.x, p.y, p.z])
QuaternionToArr = lambda p: np.array([p.x, p.y, p.z, p.w])


def get_poses_from_grasps(grasping_poses):
    poses = [a.pose for a in grasping_poses.grasps]

    for i in range(len(poses)):
        point = get_proj_point(grasping_poses.grasps[i].pose, grasping_poses.grasps[i].contact_point)
        # poses[i].position = grasping_poses.grasps[i].contact_point
        poses[i].position = point
 
        _, _, z = euler_from_quaternion(QuaternionToArr(poses[i].orientation))
        if (z > 90 * math.pi / 180) or (z < -90 * math.pi / 180):
            z = math.pi - z
            
        new_quat = quaternion_from_euler(0, 0, z)

        poses[i].orientation = ArrayToRos(Quaternion, new_quat)

    # print([a.score for a in grasping_poses.grasps])
    poses = poses[::-1]


    br.sendTransform(PointToArr(poses[0].position),
                    QuaternionToArr(poses[0].orientation),
                    rospy.Time.now(),
                    f'grasp_pose',
                    "camera_color_optical_frame")
    return poses


class AttemptCounter:
    def __init__(self):
        self.val = 0

    def update(self):
        self.val += 1

    def reset(self):
        self.val = 0


def get_points_with_vectors(obj_coordinates, number_of_points, zmax=cfg.spiral_z_max, zmin=cfg.spiral_z_min, max_radius=cfg.max_radius):
    x0, y0, z0 = obj_coordinates

    # z = np.linspace(zmin, zmax, number_of_points)[::-1]
    # np.linspace()

    base = 150
    
    z = np.logspace(math.log(zmin, base), math.log(zmax, base), number_of_points, base=base)#[::-1]
    # print(z)
    x = max_radius * (1 - (z - zmin) / (zmax - zmin)) * np.sin(cfg.frec_mult * (z - zmin)) + x0
    y = max_radius*(1 - (z - zmin) / (zmax - zmin)) * np.cos(cfg.frec_mult * (z - zmin)) + y0
    # x = max_radius*(1 / z)*np.sin(cfg.frec_mult*z) + x0
    # y = max_radius*(1 / z)*np.cos(cfg.frec_mult*z) + y0

    u = x0 - x
    v = y0 - y
    w = z0 - z

    return np.array([x, y, z]).T, np.array([u, v, w]).T


def get_center_coordinates(depth_ros):
    depth = bridge.imgmsg_to_cv2(depth_ros, desired_encoding='16UC1')

    K = rospy.wait_for_message(
        '/camera/aligned_depth_to_color/camera_info', CameraInfo).K

    K = np.array(K).reshape((3, 3))

    thresh = (depth > 0).astype(np.uint8)
    thresh[thresh > 0] = 255
    thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, np.ones((5, 5)))

    cntrs = cv.findContours(thresh, cv.RETR_EXTERNAL,
                           cv.CHAIN_APPROX_SIMPLE)[0]

    if len(cntrs) == 0:
        return ArrayToRos(Point, cfg.initial_box_coords)
    lengths = []
    for cnt in cntrs:
        lengths.append(len(cnt))
    cntr = cntrs[np.argmax(lengths)]

    
    M = cv.moments(cntr)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    z = np.float32(depth[cY, cX]) / 1000
    x = (cX - K[0, 2]) * z / K[0, 0]
    y = (cY - K[1, 2]) * z / K[1, 1]

    ret_val = Point(x=x, y=y, z=z)
    return ret_val



def get_proj_point(grasp_pose, cp):
    gp = PointToArr(grasp_pose.position)
    cp = PointToArr(cp)
    quat = QuaternionToArr(grasp_pose.orientation)

    vector = quaternion_matrix(quat)[:-1, 2]

    v1 = np.array(vector, dtype=np.float32)
    v2 = np.array(cp - gp, dtype=np.float32)
    p3 = gp + np.dot(v2, v1) / np.dot(v1, v1) * v1

    return Point(*p3)


def get_rotation_by_vector(vector):

    if vector[2] == 0:
        vector[2] = 1e-10

    z = vector / np.linalg.norm(vector)
    y = np.array([-1, 0, z[0] / z[2]])
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    x /= np.linalg.norm(x)

    R_matrix = np.eye(4)
    R_matrix[:-1, :-1] = np.array([x, y, z]).T

    ret = quaternion_from_matrix(R_matrix)
    return ret


def get_list_of_poses(obj_position_in_camera_frame, shift=0.05, camera_frame='iiwa_link_ee_camera', base_frame='world'):
    listener = tf.TransformListener()

    # camera_to_gripper_transform = np.eye(4)
    # camera_to_gripper_transform[:-1,-1] = np.array([0, 0.06, 0.12])
    

    p_shifted = np.array(
        [[0, 0, 0, 1],
        # [0 + shift, 0, 0, 1],
        # [0, 0 + shift, 0, 1],
        [0, 0, 0 + shift, 1],
        [0, 0, 0 + 2*shift, 1],
        # [0 - shift, 0, 0, 1],
        # [0, 0 - shift, 0, 1],
        [0, 0, 0 - shift, 1]])

    # p_shifted = [[*obj_position_in_camera_frame[0:-1], 0, 1]]

    for i in range(len(p_shifted)):
        p_shifted[i] += np.array([*obj_position_in_camera_frame[0:-1], 0, 0])
        p_shifted[i] += np.array([0, 0.06, 0.12, 0])

    # print(p_shifted)
    listener.waitForTransform(base_frame, camera_frame, rospy.Time(0), rospy.Duration(secs=5))
    (trans,rot) = listener.lookupTransform(base_frame, camera_frame, rospy.Time(0))

    H = quaternion_matrix(rot)
    H[:-1, -1] = trans
    # obj_p_world = quaternion_matrix(rot) @ obj_position_in_camera_frame + trans

    obj_p_world = H @ np.array([*obj_position_in_camera_frame, 1])

    camera_yi = H[0, 1]
    camera_yj = H[1, 1]

    poses = []
    for p in p_shifted:

        # p = (camera_to_gripper_transform @ p)
        
        p_world = (H @ p)
        z = (obj_p_world - p_world)[:-1]
        if z[2] == 0:
            z[2] = 1e-6
        if z[1] == 0:
            z[1] = 1e-6

        y = np.array([camera_yi, camera_yj, - (camera_yi * z[0] + camera_yj * z[1]) / z[2]])
        # y = np.array([1, 0, -z[0] / z[2]])
        x = np.cross(y, z)

        # x = np.array([1, -z[0] / z[1], 0])
        # y = np.cross(z, x)

        x /= np.linalg.norm(x)
        y /= np.linalg.norm(y)
        z /= np.linalg.norm(z)

        R_matrix = np.eye(4)
        R_matrix[:-1, :-1] = np.array([x, y, z]).T
        # quat = quaternion_from_matrix(R_matrix)
        quat = np.array([1, 0, 0, 0])

        poses.append((p_world, quat))
        
    return poses

def check_service(fail_return_value):
    def check_service_with_value(function):
        def wrapper(*args, **kwargs):
            try: 
                
                return function(*args, **kwargs)
            except rospy.ROSException:
                return fail_return_value
        return wrapper
    return check_service_with_value


if __name__ == '__main__':

    # ret = get_points(1, 1, 0, 5)
    # ret.shape

    # print(get_rotation_by_vector(np.array((0, 0.1, 0.5), dtype=np.float32))
    #

    # points, vectors = get_points_with_vectors(
    #     cfg.plane_obj_coordinates, cfg.learn_n_points)
    # # 2.5 check if all points are achievable ?

    # # 3. Move and save frames

    # rospy.init_node('my_tf_broadcaster')
    # br = tf.TransformBroadcaster()
    # rate = rospy.Rate(10.0)

    # while not rospy.is_shutdown():
    #     for idx, (point, vector) in enumerate(zip(points, vectors)):

    #         quat = get_rotation_by_vector(vector)
    #         print(vector)

    #         br.sendTransform(point,
    #                          quat,
    #                          rospy.Time.now(),
    #                          f'{idx}',
    #                          "world")
    #         rate.sleep()

    rospy.init_node("blabla", log_level=rospy.INFO)

    br = tf.TransformBroadcaster()

    pos_array = [2, 0, 0]
    or_array = quaternion_from_euler(0, 0, 0)
    p2_array = [3, 0, 3]
    center_point = np.array(PointToArr(get_proj_point(rosPose(pos_array, or_array), ArrayToRos(Point, p2_array))), dtype=np.float32)


    while True:
        br.sendTransform(pos_array,
                        or_array,
                        rospy.Time.now(),
                        'gp',
                        "camera_color_optical_frame")
        
        br.sendTransform(p2_array,
                        or_array,
                        rospy.Time.now(),
                        'cp',
                        "camera_color_optical_frame")

        br.sendTransform(center_point,
                        or_array,
                        rospy.Time.now(),
                        'center_point',
                        "camera_color_optical_frame")
        print('a')
