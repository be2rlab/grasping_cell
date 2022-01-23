import cv2 as cv
import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.structures import ImageList
from detectron2.modeling import build_model
from detectron2.structures import Instances
from detectron2.checkpoint import DetectionCheckpointer
# from detectron2.export import TracingAdapter
# from detectron2.export.flatten import flatten_to_tuple


import torch
# import cv2 as cv
import numpy as np
# from detectron2.layers.nms import batched_nms
from torchvision.ops import nms
import rospy
import os
from pathlib import Path

import cv2 as cv
from matplotlib import pyplot as plt

import yaml
import time

from utilities.utils import get_padded_image


class MaskRCNNWrapper:
    def __init__(self):
        self.cfg = get_cfg()

        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        self.cfg.merge_from_file(model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

        # self.cfg.merge_from_file('model_config.yaml')

        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.01  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        # self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        #     "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

        # username = pwd.getpwuid( os.getuid() )[ 0 ]

        new_dir = '/ws/src/grasping_vision/scripts'
        rospy.logwarn(f'current dir: {os.getcwd()}')
        try:
            os.chdir(new_dir)
        except Exception as e:
            print('Working dir not changed: ', e)
        rospy.logwarn(
            f'No model weights found in {os.getcwd()}, downloading...')
        if not Path('model_final_f10217.pkl').is_file():
            os.system(
                'wget https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl')

        self.cfg.MODEL.WEIGHTS = 'model_final_f10217.pkl'

        # with open('model_config.yaml', 'w') as fp:
        #     yaml.dump(self.cfg, fp)

        self.model = build_model(self.cfg)
        self.model.eval()
        if torch.cuda.is_available():
            self.model.cuda()

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(self.cfg.MODEL.WEIGHTS)

    def __call__(self, image):

        t_image = ImageList.from_tensors(
            [torch.Tensor(image).permute(2, 0, 1)])

        t_image_cuda = t_image.tensor.cuda()

        features = self.model.backbone(
            t_image_cuda)

        start_0 = time.time()

        proposals, _ = self.model.proposal_generator(
            t_image, features)

        min_idx = -1
        ids = []
        for idx, (box, score) in enumerate(zip(proposals[0].proposal_boxes, torch.sigmoid(proposals[0].objectness_logits))):

            if score < 0.80:
                break

            pts = box.detach().cpu().long()

            # if area is too big or if confidence is less than threshold
            if ((pts[2] - pts[0]) * (pts[3] - pts[1]) / (image.shape[0]*image.shape[1]) > 0.3):
                continue

            ids.append(idx)

        inds_after_nms = nms(
            proposals[0].proposal_boxes[ids].tensor.cpu(), proposals[0].objectness_logits[ids].cpu(), 0.1)

        new_prop = proposals[0][ids][inds_after_nms]

        instances = self.model.roi_heads(
            t_image, features, [new_prop])
        instances = instances[0][0]

        insts_inds_after_nms = nms(
            instances.pred_boxes.tensor, instances.scores, 0.3)
        instances = instances[insts_inds_after_nms]

        cropped_objects = []

        # final_masks = []
        masks = []
        for box, mask in zip(instances.pred_boxes, instances.pred_masks):
            # get masked images
            x1, y1, x2, y2 = box.round().long()
            sz = (int(x2 - x1), int(y2 - y1))

            mask_rs = cv.resize(
                mask.squeeze().detach().cpu().numpy(), sz)

            cur_mask = np.zeros((image.shape[:-1]), dtype=np.uint8)
            cur_mask[y1:y2, x1:x2] = (mask_rs + 0.5).astype(int)

            image_masked = cv.bitwise_and(image, image, mask=cur_mask)

            final_mask = image_masked[y1:y2, x1:x2]

            final_mask_sq = get_padded_image(final_mask)
            if cur_mask.max() == 0:
                print()
            masks.append(cur_mask)
            cropped_objects.append(final_mask_sq)

        return cropped_objects, np.array(masks)


if __name__ == '__main__':

    # serialize model parts

    net = MaskRCNNWrapper()

    # traced_backbone = net.model.backbone

    image = torch.ones([480, 640, 3])
    t_image = ImageList.from_tensors(
        [image.permute(2, 0, 1)])

    # deployed_backbone = torch.jit.trace(net.model.backbone, t_image.tensor.cuda(), strict=False)

    # net.forward(image)
    feats = net.model.backbone(t_image.tensor.cuda())
    # torch.save(deployed_backbone, 'traced_backbone.pth')

    # features = deployed_backbone(t_image.tensor.cuda())

    # adapter = TracingAdapter(net.model.proposal_generator, [t_image, features])
    # deployed_RPN = torch.jit.trace(adapter, adapter.flattened_inputs, strict=False)
