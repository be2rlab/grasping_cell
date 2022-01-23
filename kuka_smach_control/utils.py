from operator import inv
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import Pose, Point
import cv2 as cv
from pyquaternion import Quaternion
import config as cfg

bridge = CvBridge()

class AttemptCounter:
    def __init__(self):
        self.val = 0
        
    def update(self):
        self.val += 1
        
    def reset(self):
        self.val = 0
        

def get_points_with_vectors(obj_coordinates, number_of_points, zmax=cfg.spiral_z_max, zmin=cfg.spiral_z_min, max_radius=cfg.max_radius):
    x0, y0, z0 = obj_coordinates

    z = np.linspace(zmin, zmax, number_of_points)[::-1]
    x = max_radius*(1-z)*np.sin(20*z) + x0
    y = max_radius*(1-z)*np.cos(20*z) + y0

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

    cntr = cv.findContours(thresh, cv.RETR_EXTERNAL,
                           cv.CHAIN_APPROX_SIMPLE)[0]

    M = cv.moments(cntr[0])
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    z = np.float32(depth[cY, cX]) / 1000
    x = (cX - K[0, 2]) * z / K[0, 0]
    y = (cY - K[1, 2]) * z / K[1, 1]

    ret_val = Pose()
    ret_val.position.x = x
    ret_val.position.y = y
    ret_val.position.z = z

    return ret_val


def get_proj_point(grasp_pose, cp):

    gp = [grasp_pose.position.x,
          grasp_pose.position.y, grasp_pose.position.z]

    cp = [cp.x,
          cp.y, cp.z]
    quat = [grasp_pose.orientation.x, grasp_pose.orientation.y,
            grasp_pose.orientation.z, grasp_pose.orientation.w]
    quat = Quaternion(*quat)
    ps = np.array([gp, cp], dtype=np.float32)
    vector = quat.get_axis()
    v1 = np.array(vector, dtype=np.float32)

    v2 = np.array(ps[1, :] - ps[0, :], dtype=np.float32)

    p3 = ps[1, :] + np.dot(-v2, v1) / np.dot(v1, v1) * v1

    return Point(*p3)


def get_rotation_by_vector(vector):

    if vector[1] == 0:
        vector[1] = 1e-10

    # vector /= np.sum(np.power(vector, 2))
    # vector = normalize([vector])[0]
    z = vector / np.linalg.norm(vector)
    # z = np.ceil(z).astype(np.int32)
    y = np.array([1, -z[0] / z[1], 0])
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    x /= np.linalg.norm(x)

    # x /= np.sum(x)

    R_matrix = np.eye(4)
    R_matrix[:-1,:-1] = np.array([x, y, z]).T


    ret = quaternion_from_matrix(R_matrix)
    return ret


import rospy
import tf
from tf.transformations import quaternion_from_matrix

if __name__ == '__main__':

    # ret = get_points(1, 1, 0, 5)
    # ret.shape

    # print(get_rotation_by_vector(np.array((0, 0.1, 0.5), dtype=np.float32))
    # 

    points, vectors = get_points_with_vectors(cfg.plane_obj_coordinates, cfg.learn_n_points)
    # 2.5 check if all points are achievable ?

    # 3. Move and save frames

    rospy.init_node('my_tf_broadcaster')
    br = tf.TransformBroadcaster()
    rate = rospy.Rate(10.0)


    
    while not rospy.is_shutdown():
        for idx, (point, vector) in enumerate(zip(points, vectors)):


            quat = get_rotation_by_vector(vector)
            print(quat)

            br.sendTransform(point,
                            quat,
                            rospy.Time.now(),
                            f'{idx}',
                            "camera_color_optical_frame")
            rate.sleep()


