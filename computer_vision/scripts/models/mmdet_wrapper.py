
import numpy as np
from mmdet.apis import inference_detector, init_detector
import cv2 as cv

class MMDetWrapper:
    def __init__(self, segm_conf_thresh=0.7, segm_config='models/mmdet_config.py', segm_checkpoint='models/latest.pth', **kwargs):

        self.conf_thresh = segm_conf_thresh
        # initialize the detector
        self.model = init_detector(
            segm_config, segm_checkpoint, device='cuda:0')

    def __call__(self, image):


        result = inference_detector(self.model, image)

        cropped_objects = []
        masks = []
        for box, mask in zip(result[0][0], result[1][0]):

            box, conf = box[:-1], box[-1]


            if conf < self.conf_thresh:
                continue
            if all(box == 0):
                box[0] = np.min(np.where(mask)[1])
                box[1] = np.min(np.where(mask)[0])
                box[2] = np.max(np.where(mask)[1])
                box[3] = np.max(np.where(mask)[0])
            box = box.astype(int)

            tmp_im = image.copy()


            tmp_im[~mask] = (0, 0, 0)

            tmp_im = tmp_im[box[1]: box[3], box[0]:box[2]]

            # plt.imshow(tmp_im)
            # plt.show()
            mask = cv.morphologyEx(mask.astype(np.uint8), cv.MORPH_ERODE, np.ones(
                (3, 3), np.uint8)).astype(np.uint8)
            masks.append(mask)

            cropped_objects.append(tmp_im)

        return cropped_objects, np.array(masks)
