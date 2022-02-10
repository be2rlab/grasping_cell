#!/home/server3090/anaconda3/envs/tf-gpu-example/bin/python
#!/usr/bin/python3

import os
import sys
import argparse
import cv2
from debugpy import listen

from data import depth2pc

# ROS
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
import message_filters
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo
from contact_graspnet_planner.srv import ContactGraspNetPlanner, ContactGraspNetPlannerResponse, ContactGraspNetAnswer, ContactGraspNetAnswerResponse
from contact_graspnet_planner.msg import ContactGraspVect
from contact_graspnet_planner.msg import ContactGrasp

# import glob
import copy as copy_module

# vizual
from matplotlib import pyplot as plt

# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(os.path.join(BASE_DIR))
import config_utils
from data import load_available_input_data
from contact_grasp_estimator import GraspEstimator
from visualization_utils import visualize_grasps, show_image

from std_srvs.srv import Trigger, TriggerResponse

import numpy as np
from scipy.spatial.transform import Rotation as R
import threading

lock = threading.Lock()
# freq = 100
# conf_thresh = 0.8
# min_dist_thresh = 0.2


class ImageListener:

    def __init__(self):
        self.im_ros = None
        self.im = None
        self.depth = None
        self.depth_ros = None
        self.depth_crp = None
        self.depth_crp_ros = None
        self.depth_encoding = None
        # self.seg_ros = None
        # self.seg = None
        self.segmask = Image()

        self.pc_full = None
        self.flag = 0
        self.camera_matrix_K = None
        self.cv_brdg = CvBridge()
        self.grasp_msg = ContactGraspVect()
        self.pred_grasps_cam = {}
        self.scores = {}
        self.pc_color = None
        global_config = config_utils.load_config(
            '../checkpoints/scene_test_2048_bs3_hor_sigma_001/config.yaml', batch_size=1, arg_configs=[]
        )
        self.grasp_estimator = GraspEstimator(global_config)

        # initialize a node
        rospy.init_node("grasping_generating", log_level=rospy.INFO)
        self.camera_info_ros = rospy.wait_for_message(
            '/camera/aligned_depth_to_color/camera_info', CameraInfo)
        self.rgb_pub = rospy.Publisher('/rgb', Image, queue_size=10)

        self.depth_pub = rospy.Publisher('/depth', Image, queue_size=10)

        response_service = rospy.Service(
            '/response', ContactGraspNetAnswer, self.service_handler)

    def service_handler(self, request):

        color_im = rospy.wait_for_message('/camera/color/image_raw', Image)
        depth_im = rospy.wait_for_message(
            '/camera/aligned_depth_to_color/image_raw', Image)

        masked_msg = request.mask
        masked_im = copy_module.deepcopy(
            self.cv_brdg.imgmsg_to_cv2(masked_msg, desired_encoding='8UC1'))
        masked_im[masked_im > 0] = 1

        segmask = self.cv_brdg.cv2_to_imgmsg(masked_im)

        # depth_im = copy_module.deepcopy(self.cv_brdg.imgmsg_to_cv2(depth_im))

        # depth_im[depth_im > 800] = 0
        # depth_im /= 1000

        # plt.imshow(depth_im)
        # plt.show()

        # depth_im = self.cv_brdg.cv2_to_imgmsg(depth_im)

        # request service to server
        service_name = 'grasp_planner'
        rospy.loginfo('Wait for the grasp_planner_server')
        rospy.wait_for_service(service_name)

        try:
            rospy.loginfo("Request Contact-GraspNet grasp planning")

            grasp_planner = rospy.ServiceProxy(
                service_name, ContactGraspNetPlanner)
            resp = grasp_planner(
                color_im,
                depth_im,
                self.camera_info_ros,
                segmask
            )
            # print(resp.grasps[0])
            rospy.loginfo("Get {} grasps from the server.".format(
                len(resp.grasps)))
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: {}".format(e))

        if len(resp.grasps) == 0:
            rospy.logerr('No candidates generated')
            self.flag = 0
            return ContactGraspNetAnswerResponse(success=False, grasps=[])

        contact_pts = {}
        id_list = []
        grasp_list = []
        scores_list = []
        contact_pts_list = []
        for grasp in resp.grasps:
            instance_id = grasp.id
            pose_msg = grasp.pose
            score = grasp.score
            contact_pt_msg = grasp.contact_point

            # get transform matrix
            tf_mat = np.zeros((4, 4), dtype=np.float64)
            quat = [pose_msg.orientation.x, pose_msg.orientation.y,
                    pose_msg.orientation.z, pose_msg.orientation.w]
            rot_mat = R.from_quat(quat)
            tf_mat[0:3, 0:3] = rot_mat.as_matrix()
            tf_mat[0, 3] = pose_msg.position.x
            tf_mat[1, 3] = pose_msg.position.y
            tf_mat[2, 3] = pose_msg.position.z

            # get contact point as numpy
            contact_pt = np.array(
                [contact_pt_msg.x, contact_pt_msg.y, contact_pt_msg.z])

            # append to list
            id_list.append(instance_id)
            grasp_list.append(tf_mat)
            scores_list.append(score)
            contact_pts_list.append(contact_pt)

        # convert list to numpy array
        id_list = np.array(id_list)
        grasp_list = np.array(grasp_list)
        scores_list = np.array(scores_list)
        contact_pts_list = np.array(contact_pts_list)

        # put on the dictionary
        for instance_id in id_list:
            indices = np.where(id_list == instance_id)[0]
            self.pred_grasps_cam[instance_id] = grasp_list[indices]
            self.scores[instance_id] = scores_list[indices]
            contact_pts[instance_id] = contact_pts_list[indices]

            # make ros msg of top 5 highest score grasps (pose,score,contact points, id)

        self.grasp_msg = ContactGraspVect()
        for i, k in enumerate(self.pred_grasps_cam):
            largest_scores_ind = np.argsort(self.scores[k])
            top5 = largest_scores_ind[-5:]
            for j in range(len(top5)):
                self.grasp_msg.grasps_vect.append(
                    self.make_grasp_msg(
                        self.pred_grasps_cam[k][top5[j]],
                        self.scores[k][top5[j]],
                        contact_pts[k][top5[j]],
                        top5[j]
                    )
                )
        # self.flag = 1

        # plt.imshow(masked_depth_im)
        # plt.show()

        self.pc_full, pc_segments, self.pc_color = self.grasp_estimator.extract_point_clouds(
            depth=self.cv_brdg.imgmsg_to_cv2(depth_im),
            K=np.array(self.camera_info_ros.K).reshape((3, 3)),
            rgb=self.cv_brdg.imgmsg_to_cv2(color_im),
            skip_border_objects=False,
            z_range=[200, 1800],
        )

        self.pc_full /= 1000

        # # show_image(listener.im, None)

        self.flag = True

        return ContactGraspNetAnswerResponse(success=True, grasps=self.grasp_msg.grasps_vect)

    def make_grasp_msg(self, se3, score, contact_point, ind):
        msg = ContactGrasp()
        msg.score = score
        msg.id = ind
        msg.contact_point.x = contact_point[0]
        msg.contact_point.y = contact_point[1]
        msg.contact_point.z = contact_point[2]

        r = R.from_matrix(se3[0:3, 0:3])
        quat = r.as_quat()  # quat shape (4,)

        point = se3[0:3, 3]

        msg.pose.position.x = point[0]
        msg.pose.position.y = point[1]
        msg.pose.position.z = point[2]
        msg.pose.orientation.x = quat[0]
        msg.pose.orientation.y = quat[1]
        msg.pose.orientation.z = quat[2]
        msg.pose.orientation.w = quat[3]

        return msg


if __name__ == '__main__':

    listener = ImageListener()
    rate = rospy.Rate(100)

    rospy.loginfo('init complete')
    # print(listener.flag)
    while not rospy.is_shutdown():
        if listener.flag == True:

            visualize_grasps(listener.pc_full, listener.pred_grasps_cam,
                             listener.scores, plot_opencv_cam=True, pc_colors=listener.pc_color)

            listener.flag = False

    rospy.spin()
