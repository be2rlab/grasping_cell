#!/usr/bin/python3.8


#!/home/iiwa/anaconda3/envs/uoais2/bin/python


import time
import threading

import numpy as np
import message_filters
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2 as cv
from segmentation.msg import SegmentAndClassifyResult
from segmentation.srv import SegmentAndClassifyService, SegmentAndClassifyServiceResponse

from utilities.utils import find_nearest_mask
# sys.path.insert(0, '/home/iiwa/Nenakhov/segm_ros_ws/src/segmentation')
import os


lock = threading.Lock()


class FilteringNode:
    def __init__(self):
        # type 'topic' or 'service'
        self.type = 'topic'
        self.cv_bridge = CvBridge()

        # initialize a node
        rospy.init_node("crop_depth_node", log_level=rospy.INFO)

        self.base_frame = 'measured/base_link'
        self.camera_frame = 'measured/camera_color_optical_frame'
        self.target_frame = self.base_frame

        if self.type == 'topic':
            self.res_pub = rospy.Publisher(
                '/depth_masked', Image, queue_size=10)

            self.depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw',
                                                        Image, queue_size=10)

            self.segm_results_sub = message_filters.Subscriber('/segm_results',
                                                               SegmentAndClassifyResult, queue_size=10)

            ts = message_filters.ApproximateTimeSynchronizer(
                [self.depth_sub, self.segm_results_sub], 1, 0.1)
            ts.registerCallback(self.callback_sub)

        # elif self.type == 'service':
        #     self.srv = rospy.Service(
        #         'segmentation_service_node', SegmentAndClassifyService, self.service_callback)
            rospy.loginfo('Service is running')

        self.depth = None
        self.masks = None
        self.rgb_msg_header = None

        print('Init complete!')

    def service_callback(self, msg):
        rgb_msg = rospy.rgb_sub.wait_for_message(self.rgb_sub, Image)
        depth_msg = rospy.depth_sub.wait_for_message(self.depth_sub, Image)

        rgb_im, depth_im = self.convert_msgs_to_images(rgb_msg, depth_msg)

        return self.run_segmentation(rgb=rgb_im, depth=depth_im)

    def callback_sub(self, depth_msg, segm_results_msg):
        depth_im = self.cv_bridge.imgmsg_to_cv2(
            depth_msg, desired_encoding='32FC1')

        masks_imgs = [self.cv_bridge.imgmsg_to_cv2(
            mask_msg, desired_encoding='mono8') for mask_msg in segm_results_msg.masks]

        with lock:
            self.depth = depth_im.copy()
            self.seg_results = segm_results_msg
            self.masks = masks_imgs

    def process(self, depth_im=None, masks_imgs=None):

        if depth_im is None and masks_imgs is None:
            with lock:
                if self.depth is None:
                    rospy.logerr_throttle(5, "No image received")
                    return
                depth = self.depth.copy()
                masks = self.masks.copy()

        nearest_mask = find_nearest_mask(depth, masks)

        if self.type == 'topic':
            nearest_mask_msg = self.cv_bridge.cv2_to_imgmsg(
                nearest_mask.astype(np.uint8), encoding='mono8')

            nearest_mask_msg.header.stamp = rospy.get_rostime()

            self.res_pub.publish(nearest_mask_msg)
        # elif self.type == 'service':
        #     return SegmentAndClassifyServiceResponse(results)


if __name__ == '__main__':
    node = FilteringNode()
    rate = rospy.Rate(1000)
    if node.type == 'topic':
        while not rospy.is_shutdown():
            node.process()
            rate.sleep()
    else:
        rospy.spin()
