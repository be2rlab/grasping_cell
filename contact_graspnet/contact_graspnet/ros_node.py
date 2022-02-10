#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import time
import glob

import numpy as np
from scipy.spatial.transform import Rotation as R

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR))
import config_utils
from contact_grasp_estimator import GraspEstimator

import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from contact_graspnet_planner.msg import ContactGrasp
from contact_graspnet_planner.srv import ContactGraspNetPlanner
from contact_graspnet_planner.srv import ContactGraspNetPlannerResponse
from visualization_utils import visualize_grasps

def imgmsg_to_cv2(img_msg):
    """Convert ROS Image messages to OpenCV images.

    `cv_bridge.imgmsg_to_cv2` is broken on the Python3.
    `from cv_bridge.boost.cv_bridge_boost import getCvType` does not work.

    Args:
        img_msg (`sonsor_msgs/Image`): ROS Image msg

    Raises:
        NotImplementedError: Supported encodings are "8UC3" and "32FC1"

    Returns:
        `numpy.ndarray`: OpenCV image
    """
    # check data type

    if img_msg.encoding == '8UC3':
        dtype = np.uint8
        n_channels = 3
    elif img_msg.encoding == '32FC1':
        dtype = np.float32
        n_channels = 1
    elif img_msg.encoding == 'rgb8':
        dtype = np.uint8
        n_channels = 3
        # img_msg = CvBridge.imgmsg_to_cv2(img_msg, desired_encoding='8UC3')
    elif img_msg.encoding == '16UC1':
        # img_msg = CvBridge.imgmsg_to_cv2(img_msg, desired_encoding='32FC1')
        dtype = np.float32
        n_channels = 1
        res = cv2.normalize(img_msg, res, 0, 1, cv2.NORM_MINMAX)
        img_msg = res
    elif img_msg.encoding == '8UC1':
        dtype = np.uint8
        n_channels = 1
    else:
        raise NotImplementedError('custom imgmsg_to_cv2 does not support {} encoding type'.format(img_msg.encoding))

    # bigendian
    dtype = np.dtype(dtype)
    dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')
    if n_channels == 1:
        img = np.ndarray(shape=(img_msg.height, img_msg.width),
                         dtype=dtype, buffer=img_msg.data)
    else:
        img = np.ndarray(shape=(img_msg.height, img_msg.width, n_channels),
                         dtype=dtype, buffer=img_msg.data)

    # If the byt order is different between the message and the system.
    if img_msg.is_bigendian == (sys.byteorder == 'little'):
        img = img.byteswap().newbyteorder()
    return img


class GraspPlannerServer(object):
    def __init__(self, global_config, checkpoint_dir, local_regions=True, skip_border_objects=False, filter_grasps=True, segmap_id=None, z_range=[0.2,1.8], forward_passes=1):
        # get parameters
        self.local_regions = local_regions
        self.skip_border_objects = skip_border_objects
        self.filter_grasps = filter_grasps
        self.segmap_id = segmap_id
        self.z_range = z_range
        self.forward_passes = forward_passes

        # Build the model
        self.grasp_estimator = GraspEstimator(global_config)
        self.grasp_estimator.build_network()

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver(save_relative_paths=True)

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        self.sess = tf.Session(config=config)

        # Load weights
        self.grasp_estimator.load_weights(self.sess, saver, checkpoint_dir, mode='test')

        # ros cv bridge
        self.cv_bridge = CvBridge()

        # ros service
        rospy.Service("grasp_planner", ContactGraspNetPlanner, self.plan_grasp_handler)
        rospy.loginfo("Start Contact-GraspNet grasp planner.")


        self.pc_full = None
        self.pred_grasps_cam = None
        self.scores = None
        self.pc_colors = None
        self.flag = None

    def plan_grasp_handler(self, req):
        #############
        # Get Input #
        #############
        # exmaple data format get input from npy
        # pc_segments = {}
        # segmap, rgb, depth, cam_K, pc_full, pc_colors = load_available_input_data(p, K=K)
        # segmap : (720, 1280), [0,12]
        # rgb : (720, 1280, 3)
        # cam_K : fx=912.72143555, fy=912.7409668, cx-649.00366211, cy=363.2547192
        # pc_full : None
        # pc_colors : None

        # unpack request massage
        color_im, depth_im, segmask, camera_intr = self.read_images(req)
        # print(segmask)
        # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        # print(np.size(segmask))
        
        if np.mean(segmask) == -1: #or np.size(segmask) == 1:
            segmask = None
            rospy.logwarn('No segmentation mask. Generate grasp with full scene.')
            if (self.local_regions or self.filter_grasps):
                rospy.logerr('For the invalid segmentation mask, local_regions or filter_grasp should be False.')
        else:
            rospy.loginfo('Num instance id: {}, filter_grasp: {}, local_region: {}'.format(
                np.max(segmask), self.filter_grasps, self.local_regions))

        # Convert depth image to point clouds
        rospy.loginfo('Converting depth to point cloud(s)...')
        
        pc_full, pc_segments, pc_colors = self.grasp_estimator.extract_point_clouds(
            depth=depth_im,
            segmap=segmask,
            K=camera_intr,
            rgb=color_im,
            skip_border_objects=self.skip_border_objects,
            z_range=self.z_range,
            )


        #############
        # Gen Grasp #
        #############
        # if fc_full, key=-1
        # pred_grasps_cam : dict.keys=[1, num_instance], TF(4x4) on camera coordinate
        # scores : dict.keys=[1, num_instance]
        # contact_pts : dict.keys=[1, num_instance], c.p(3) on camera coordinate

        # Generate grasp
        start_time = time.time()
        rospy.loginfo('Start to generate grasps')
        pred_grasps_cam, scores, contact_pts, _ = self.grasp_estimator.predict_scene_grasps(
            self.sess,
            pc_full,
            pc_segments=pc_segments,
            local_regions=self.local_regions,
            filter_grasps=self.filter_grasps,
            forward_passes=self.forward_passes,
            )

        # Generate grasp responce msg
        grasp_resp = ContactGraspNetPlannerResponse()
        for instance_id in pred_grasps_cam.keys():
            grasp_score_cp = zip(pred_grasps_cam[instance_id], scores[instance_id], contact_pts[instance_id])
            for grasp, score, contact_pt in grasp_score_cp:
                grasp_msg = self.get_grasp_msg(instance_id, grasp, score, contact_pt)
                grasp_resp.grasps.append(grasp_msg)
        rospy.loginfo('Generate grasp {} took {}s'.format(len(grasp_resp.grasps), time.time() - start_time))

        # save for visualizing
        self.pc_full = pc_full
        self.pred_grasps_cam = pred_grasps_cam
        self.scores = scores
        self.pc_colors = pc_colors
        self.flag = True

        return grasp_resp

    def read_images(self, req):

        # responded with an error: b"error processing request: unsupported operand type(s) for /: 'Image' and 'int'"

        # responded with an error: b"error processing request: name 'color_im' is not defined"

        """Reads images from a ROS service request.

        Parameters
        ---------
        req: :obj:`ROS ServiceRequest`
            ROS ServiceRequest for grasp planner service.
        """
        # Get the raw depth and color images as ROS `Image` objects.
        raw_color = req.color_image
        raw_depth = req.detph_image
        raw_segmask = req.segmask

        # Get the raw camera info as ROS `CameraInfo`.
        raw_camera_info = req.camera_info

        camera_intr = np.array([raw_camera_info.K]).reshape((3, 3))
        bridge = CvBridge()
        res = None

        try:

            # color_im = imgmsg_to_cv2(raw_color)
            # depth_im = imgmsg_to_cv2(raw_depth)
            # segmask = imgmsg_to_cv2(raw_segmask)
            
            color_im = bridge.imgmsg_to_cv2(raw_color)
            # depth_im = bridge.imgmsg_to_cv2(raw_depth, '32FC1')
            segmask = bridge.imgmsg_to_cv2(raw_segmask)

            if raw_depth.encoding == '32FC1':
                depth_cv = bridge.imgmsg_to_cv2(raw_depth)
            elif raw_depth.encoding == '16UC1':
                depth_cv = bridge.imgmsg_to_cv2(
                    raw_depth).copy().astype(np.float32)
                depth_cv /= 1000.0
            else:
                rospy.logerr_throttle(
                    1, 'Unsupported depth type. Expected 16UC1 or 32FC1, got {}'.format(
                        raw_depth.encoding))
            depth_im = depth_cv.copy()
            

        except NotImplementedError as e:
            rospy.loginfo("Error cathed")
            rospy.logerr(e)

        return (color_im, depth_im, segmask, camera_intr)

    def get_grasp_msg(self, id, tf_mat, score, contact_pt):
        grasp_msg = ContactGrasp()

        # set instance id
        grasp_msg.id = id

        # convert tf matrix to pose msg
        rot = R.from_matrix(tf_mat[0:3, 0:3])
        quat = rot.as_quat()
        grasp_msg.pose.position.x = tf_mat[0, 3]
        grasp_msg.pose.position.y = tf_mat[1, 3]
        grasp_msg.pose.position.z = tf_mat[2, 3]
        grasp_msg.pose.orientation.x = quat[0]
        grasp_msg.pose.orientation.y = quat[1]
        grasp_msg.pose.orientation.z = quat[2]
        grasp_msg.pose.orientation.w = quat[3]

        # conver contact point to msg
        grasp_msg.contact_point.x = contact_pt[0]
        grasp_msg.contact_point.y = contact_pt[1]
        grasp_msg.contact_point.z = contact_pt[2]

        # get grasp msg
        grasp_msg.score = score

        return grasp_msg


if __name__ == "__main__":
    # init node
    rospy.init_node('contact_graspnet_planner')
    rospy.loginfo("Contact GraspNet Planner is launched with Python {}".format(sys.version))

    # get arguments from the ros parameter server
    ckpt_dir = rospy.get_param('~ckpt_dir')  # default='checkpoints/scene_test_2048_bs3_hor_sigma_001', help='Log dir [default: checkpoints/scene_test_2048_bs3_hor_sigma_001]'
    z_min = rospy.get_param('~z_min')
    z_max = rospy.get_param('~z_max')
    z_range = np.array([z_min, z_max])  # default=[0.2,1.8], help='Z value threshold to crop the input point cloud')
    local_regions = rospy.get_param('~local_regions')  # action='store_true', default=False, help='Crop 3D local regions around given segments.')
    filter_grasp = rospy.get_param('~filter_grasps')  # action='store_true', default=False,  help='Filter grasp contacts according to segmap.')
    skip_border_objects = rospy.get_param('~skip_border_objects')  # action='store_true', default=False,  help='When extracting local_regions, ignore segments at depth map boundary.')
    forward_passes = rospy.get_param('~forward_passes')  # type=int, default=1,  help='Run multiple parallel forward passes to mesh_utils more potential contact points.')
    segmap_id = rospy.get_param('~segmap_id')  # type=int, default=0,  help='Only return grasps of the given object id')
    # arg_configs = rospy.get_param('~arg_configs')  # nargs="*", type=str, default=[], help='overwrite config parameters')
    arg_configs = []

    # get global config
    global_config = config_utils.load_config(
        ckpt_dir,
        batch_size=forward_passes,
        arg_configs=arg_configs)

    # print config
    rospy.loginfo(str(global_config))
    rospy.loginfo('pid: %s' % (str(os.getpid())))

    # start Contact GraspNet Planner service
    planner = GraspPlannerServer(
        global_config,
        ckpt_dir,
        local_regions=local_regions,
        skip_border_objects=skip_border_objects,
        filter_grasps=filter_grasp,
        segmap_id=segmap_id,
        z_range=z_range,
        forward_passes=forward_passes)


    while not rospy.is_shutdown():
        if planner.flag == True:
            visualize_grasps(planner.pc_full, planner.pred_grasps_cam,
                             planner.scores, plot_opencv_cam=True, pc_colors=planner.pc_colors)

            planner.flag = False

    rospy.spin()
