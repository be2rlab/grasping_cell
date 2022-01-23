
import cv2 as cv
import numpy as np


from utilities.utils import get_nearest_mask_id, get_nearest_to_center_box, bb_iou

base_iou_thresh = 0.8
iou_decay = 0.05


class simple_tracker:
    def __init__(self):

        self.iou_thresh = base_iou_thresh

        self.cent_idx = None

        self.last_box = []

    def get_tracking_idx(self, boxes, image_shape, image_segmented):

        ret_image = image_segmented.copy()

        cent_idx = get_nearest_to_center_box(
            image_shape, boxes)

        ret_idx = None
        for ix, box in enumerate(boxes):

            box = box.cpu().long().numpy()
            iou_n = 0

            c = (0, 0, 255)
            if cent_idx == ix:
                if len(self.last_box) != 0:
                    iou_n = bb_iou(self.last_box, np.expand_dims(box, axis=0))

                    if iou_n > self.iou_thresh:
                        self.last_box = box

                        ret_idx = ix
                        c = (255, 0, 0)
                else:
                    self.last_box = box

            # c = (255, 0, 0) if (
            #     ix == cent_idx) and iou_n > self.iou_thresh else (0, 0, 255)

            if (ix == cent_idx) and iou_n > self.iou_thresh:
                self.counter += 1
                self.iou_thresh = self.base_iou_thresh

            if (ix == cent_idx) and iou_n < self.iou_thresh:
                self.iou_thresh -= self.iou_decay

            # draw bounding box
            pts = list(map(int, box))

            cv.rectangle(ret_image, (pts[0], pts[1]),
                         (pts[2], pts[3]), c, 2)

            # draw object masks
            # x1, y1, x2, y2 = box
            # sz = (int(x2 - x1), int(y2 - y1))

            # mask_rs = cv.resize(
            #     m.squeeze().detach().cpu().numpy(), sz)

            # cur_mask = np.zeros((image_shape[:-1]), dtype=np.uint8)
            # cur_mask[y1:y2, x1:x2] = (mask_rs + 0.5).astype(int)
            # cntrs, _ = cv.findContours(
            #     cur_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            # cv.drawContours(image_segmented, cntrs, -
            #                 1, c, 2)
        return ret_idx, ret_image


class RGBDTracker:
    def __init__(self, base_iou_thresh=0.5, iou_decay=0.05):

        self.base_iou_thresh = base_iou_thresh
        self.iou_decay = iou_decay
        self.iou_thresh = base_iou_thresh

        self.cent_idx = None

        self.last_box = []

    def get_tracking_idx(self, depth, masks, vis_image):

        nearest_mask_id = get_nearest_mask_id(depth, masks)

        # cent_idx = get_nearest_to_center_box(
        #     image_shape, boxes)

        ret_idx = None
        for ix, mask in enumerate(masks):

            box = list(cv.boundingRect(mask))
            box[2] += box[0]
            box[3] += box[1]

            iou_n = 0

            c = (0, 0, 255)
            if nearest_mask_id == ix:
                if len(self.last_box) != 0:
                    iou_n = bb_iou(self.last_box, box)
                    # iou_n = bb_iou(self.last_box, np.expand_dims(box, axis=0))

                    if iou_n > self.iou_thresh:
                        self.last_box = box

                        ret_idx = ix
                        c = (255, 0, 0)
                else:
                    self.last_box = box

            # c = (255, 0, 0) if (
            #     ix == cent_idx) and iou_n > self.iou_thresh else (0, 0, 255)

            if (ix == nearest_mask_id) and iou_n > self.iou_thresh:
                # self.counter += 1
                self.iou_thresh = base_iou_thresh

            if (ix == nearest_mask_id) and iou_n < self.iou_thresh:
                self.iou_thresh -= iou_decay

            # draw bounding box
            pts = list(map(int, box))

            cv.rectangle(vis_image, (pts[0], pts[1]),
                         (pts[2], pts[3]), c, 2)

            # draw object masks
            # x1, y1, x2, y2 = box
            # sz = (int(x2 - x1), int(y2 - y1))

            # mask_rs = cv.resize(
            #     m.squeeze().detach().cpu().numpy(), sz)

            # cur_mask = np.zeros((image_shape[:-1]), dtype=np.uint8)
            # cur_mask[y1:y2, x1:x2] = (mask_rs + 0.5).astype(int)
            # cntrs, _ = cv.findContours(
            #     cur_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            # cv.drawContours(image_segmented, cntrs, -
            #                 1, c, 2)
        return ret_idx, vis_image
