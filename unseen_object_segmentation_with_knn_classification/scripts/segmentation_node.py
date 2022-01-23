#!/home/server3090/anaconda3/envs/segmentation_ros/bin/python
#!/usr/bin/python3.8

import time
import threading

import numpy as np
# from torchvision.transforms.functional import crop
import message_filters
import rospy
from models.tracker import RGBDTracker
from std_msgs.msg import String, Header
from std_srvs.srv import Trigger, TriggerResponse
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from segmentation.msg import SegmentAndClassifyResult
from segmentation.srv import SegmentAndClassifyService, SegmentAndClassifyServiceResponse

from models.all_model import AllModel
from utilities.utils import draw_masks, find_nearest_mask, get_nearest_mask_id
# sys.path.insert(0, '/home/iiwa/Nenakhov/segm_ros_ws/src/segmentation')
import os
import torch
import cv2 as cv


lock = threading.Lock()


class VisionNode:
    def __init__(self, node_type='service'):
        # type 'topic' or 'service'
        self.type = node_type
        self.cv_bridge = CvBridge()

        self.im = None

        # initialize a node
        rospy.init_node("vision_node", log_level=rospy.INFO)

        self.base_frame = 'measured/base_link'
        self.camera_frame = 'measured/camera_color_optical_frame'
        self.target_frame = self.base_frame

        self.features_to_save = None

        self.vis_pub = rospy.Publisher(
            '/rgb_segmented', Image, queue_size=10)

        if self.type == 'topic':

            rospy.logwarn('Publisher is setting up')
            self.results_pub = rospy.Publisher(
                '/segm_results', SegmentAndClassifyResult, queue_size=10)

            self.cropped_depth_pub = rospy.Publisher(
                '/depth_masked', Image, queue_size=10)

            self.rgb_sub = message_filters.Subscriber('/camera/color/image_raw',
                                                      Image, queue_size=10)

            self.depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw',
                                                        Image, queue_size=10)

            ts = message_filters.ApproximateTimeSynchronizer(
                [self.rgb_sub, self.depth_sub], 1, 0.1)
            ts.registerCallback(self.callback_rgbd)

        elif self.type == 'service':
            self.inference_srv = rospy.Service(
                'segmentation_inference_service', SegmentAndClassifyService, self.service_inference_callback)
            rospy.loginfo('Service is setting up')


            self.im = None
            self.depth = None
            self.rgb_msg_header = None

        self.train_srv = rospy.Service(
            'segmentation_train_service', Trigger, self.service_train_callback)
            
        script_dir = os.path.dirname(os.path.realpath(__file__))

        self.model = AllModel(
            segm_config=f'{script_dir}/last_config.py',
            segm_checkpoint=f'{script_dir}/latest.pth',
            # segm_config=f'{script_dir}/config_metagraspnet.py',
            # segm_checkpoint=f'{script_dir}/best_segm_mAP_epoch_28.pth',
            segm_conf_thresh=0.8,
            fe=torch.hub.load(
                'facebookresearch/dino:main', 'dino_vits16'),
            fe_fp16=True,
            # pca_ckpt=f'{script_dir}/PCA_imagenet_160_features_vits16.pth',
            # pca_ckpt=f'{script_dir}PCA_imagenet_40_features_vits16.pth',
            # knn_file=f'{script_dir}/features_w_labels_vits16.pth',
            # knn_file=f'{script_dir}/augmented_features_w_labels_vits16.pth',
            save_to_file=False
        )

        self.tracker = RGBDTracker(base_iou_thresh=0.8, iou_decay=0.05)
        print('Init complete!')

    def convert_msgs_to_images(self, rgb_msg, depth_msg):

        depth_im = self.cv_bridge.imgmsg_to_cv2(
            depth_msg, desired_encoding='32FC1')
        rgb_im = self.cv_bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
        return rgb_im, depth_im

    def service_inference_callback(self, msg):
        rgb_msg = rospy.wait_for_message('/camera/color/image_raw', Image)
        depth_msg = rospy.wait_for_message(
            '/camera/aligned_depth_to_color/image_raw', Image)

        rgb_im, depth_im = self.convert_msgs_to_images(rgb_msg, depth_msg)

        return self.run_segmentation(rgb=rgb_im, depth=depth_im)

    def callback_rgbd(self, rgb, depth):

        rgb_im, depth_im = self.convert_msgs_to_images(rgb, depth)

        with lock:
            self.im = rgb_im.copy()
            self.depth = depth_im.copy()
            self.rgb_frame_id = rgb.header.frame_id
            self.rgb_frame_stamp = rgb.header.stamp
            self.rgb_msg_header = rgb.header

    def run_segmentation(self, rgb=None, depth=None):
        start = time.time()

        if rgb is None and depth is None:
            with lock:
                if self.im is None:
                    rospy.logerr_throttle(5, "No image received")
                    return
                rgb = self.im.copy()
                depth = self.depth.copy()

        clf_results, masks = self.model(rgb, depth)

        if clf_results[0] is None:
            rospy.logwarn('No trained classes found')
            if self.type == 'service':
                return SegmentAndClassifyServiceResponse(SegmentAndClassifyResult()) 
            else:
                return

        visualized_masks = draw_masks(
            rgb,
            depth,
            masks,
            *clf_results,
            conf_thresh=0.5,
            dist_thresh=90,
            draw_only_nearest=False,
            show_nearest=True,
            classes_list=self.model.classifier.classes,
            show_low_prob=True)

        cl_names, cl_confs, cl_dists = clf_results
        nearest_mask = get_nearest_mask_id(depth, masks)

        # prepare an answer
        results = SegmentAndClassifyResult()
        results.header = Header()

        results.mask = self.cv_bridge.cv2_to_imgmsg(
            masks[nearest_mask].astype(np.uint8), encoding='mono8')

        depth_cropped = cv.bitwise_and(
            depth, depth, mask=masks[nearest_mask])

        results.masked_depth = self.cv_bridge.cv2_to_imgmsg(
            depth_cropped.astype(np.uint16), encoding='16UC1')
        if len(cl_names) != 0:
            results.class_name = cl_names[nearest_mask]
            results.class_dist = cl_dists[nearest_mask]
            results.class_conf = cl_confs[nearest_mask]
            depth_cropped = np.zeros_like(depth)
        else:
            results.class_name = 'None'
            results.class_dist = 0.0
            results.class_conf = 0.0

        results.header.stamp = rospy.get_rostime()

        vis_msg = self.cv_bridge.cv2_to_imgmsg(
            visualized_masks.astype(np.uint8), encoding='bgr8')
        vis_msg.header.stamp = rospy.get_rostime()
        self.vis_pub.publish(vis_msg)

        rospy.loginfo_throttle(2, f'FPS: {(1 /(time.time() - start)):.2f}')
        # send an answer
        if self.type == 'topic':
            self.results_pub.publish(results)
            depth_cropped = cv.bitwise_and(
                depth, depth, mask=masks[nearest_mask])
            depth_cropped_msg = self.cv_bridge.cv2_to_imgmsg(
                depth_cropped.astype(np.uint16), encoding='16UC1')
            self.cropped_depth_pub.publish(depth_cropped_msg)
        else:
            return SegmentAndClassifyServiceResponse(results)


    def service_train_callback(self, request):

        rgb_msg = rospy.wait_for_message('camera/color/image_raw', Image)
        depth_msg = rospy.wait_for_message(
            '/camera/aligned_depth_to_color/image_raw', Image)

        rgb, depth = self.convert_msgs_to_images(rgb_msg, depth_msg)

        resp = TriggerResponse()

        masks, features, cropped_objs = self.model(rgb, depth, train=True)

        if features is None:
            resp.success = False
            resp.message = 'No masks found'
            return resp

        # ret_idx, rgb = self.tracker.get_tracking_idx(depth, masks, rgb)

        nearest_mask_id = get_nearest_mask_id(depth, masks)

        cntr = cv.findContours(
            masks[nearest_mask_id], cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]

        cl = f'object_{len(self.model.classifier.classes)}'
        fname = f'scripts/segmentation_dataset/{cl}/{(time.time()):.2f}.png'

        os.makedirs(f'segmentation_dataset/{cl}', exist_ok=True)
        cv.imwrite(fname, cropped_objs[nearest_mask_id])

        cv.drawContours(rgb, cntr, -1, (255, 0, 0), 4)
        # cv.imshow('im', rgb)

        vis_msg = self.cv_bridge.cv2_to_imgmsg(
            rgb.astype(np.uint8), encoding='bgr8')
        vis_msg.header.stamp = rospy.get_rostime()
        self.vis_pub.publish(vis_msg)

        ret = self.model.save_feature(features[nearest_mask_id])

        if ret:
            resp.success = True
            resp.message = 'kNN updated'
            # self.model.classifier.print_info()
        else:
            resp.success = True
            resp.message = 'feature saved'

        return resp


if __name__ == '__main__':
    visionNone = VisionNode()
    rate = rospy.Rate(1000)

    # visionNone.train()
    if visionNone.type == 'topic':
        while not rospy.is_shutdown():

            visionNone.run_segmentation()
            rate.sleep()
    else:
        rospy.spin()
