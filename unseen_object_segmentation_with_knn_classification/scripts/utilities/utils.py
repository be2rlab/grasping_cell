
import numpy as np
import cv2 as cv
from scipy.spatial.distance import euclidean, cosine
from matplotlib import pyplot as plt

import torch
from numpy import random

np.random.seed(0)
color_list = np.random.randint(0, 255, size=(100, 3))
# color_list = np.array([[141,  48,  66],
#    [118,  56, 148],
#    [217,  32,  14],
#    [145,  67, 157],
#    [106, 248, 210],
#    [213, 101, 104],
#    [58,  16, 121],
#    [170, 152, 245],
#    [78, 100, 118],
#    [44,  12, 231],
#    [29, 214, 172],
#    [91, 157, 128],
#    [194, 168,  24],
#    [208,  68, 112],
#    [228, 161, 228],
#    [246, 233, 241],
#    [66, 141, 171],
#    [251, 221, 124],
#    [151, 221, 165],
#    [228, 178,   8],
#    [205,  38, 134],
#    [44,  11, 157],
#    [157,  76, 250],
#    [7,  89,  75],
#    [120, 251,  37],
#    [112, 139, 189],
#    [91, 247,  82],
#    [149, 236,  24],
#    [35, 134,  99],
#    [214, 243, 180],
#    [118,  54, 191],
#    [199,  12, 123],
#    [218,  68,   2],
#    [79, 127,  14],
#    [109,  90, 42],
#    [27,  40, 144],
#    [36, 177, 152],
#    [16, 254, 246],
#    [27,  45, 250],
#    [99, 186,  61],
#    [11,  19, 154],
#    [148, 210,  42],
#    [27,  40, 144],
#    [36, 177, 152],
#    [16, 254, 246],
#    [27,  45, 250]])


def find_nearest_mask(depth, masks):
    dists = []
    if len(masks) == 0:
        return

    for mask in masks:

        cntrs, hierarchy = cv.findContours(
            mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # print(cntrs[0])
        M = cv.moments(cntrs[0])
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        dists.append(depth[cY, cX])
    ret_im = depth.copy()
    ret_im[~masks[np.argmin(dists)]] = 0
    return ret_im


def draw_masks(inp_im, depth, masks, clss, confs, dists, show_low_prob=True, show_nearest=False, conf_thresh=0.7, dist_thresh=70, draw_only_nearest=True, classes_list=None):
    # draw masks and nearest object
    image = inp_im.copy()

    if len(masks) == 0:
        return image

    if clss is None:
        clss = [None] * len(masks)
        confs = [np.nan] * len(masks)
        dists = [np.nan] * len(masks)

    depth_dists = []

    vis_cntrs_data = []
    for mask, cls, conf, dist in zip(masks.astype(np.uint8), clss, confs, dists):

        contours, hierarchy = cv.findContours(
            mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            continue

        M = cv.moments(contours[0])
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        if depth[cY, cX] != 0:
            depth_dists.append(depth[cY, cX])
        else:
            depth_dists.append(1e4)

        xmin = np.min(contours[0][:, 0, 0])
        ymin = np.min(contours[0][:, 0, 1])

        if classes_list is None:
            # color = (random.randint(0, 256), random.randint(
            #     0, 256), random.randint(0, 256))
            color = (255, 0, 0)
        elif conf <= conf_thresh or dist >= dist_thresh:
            color = (0, 0, 0)

        else:
            color = color_list[classes_list.index(cls)]

        # color = (0, 0, 255)
        # if cls is not None:
        #     if conf <= conf_thresh or dist >= dist_thresh:
        #         if show_low_prob:
        #             color = (0, 0, 0)
        #         else:
        #             continue

        text = f'{cls} {conf:.2f} {dist:.2f}'

        vis_cntrs_data.append((contours,
                               text if cls is not None else None,
                               (xmin, ymin),
                               color)
                              )

    for idx, vis_cntr_data in enumerate(vis_cntrs_data):
        if idx != np.argmin(depth_dists) and draw_only_nearest:
            continue
        cntr, text, (xmin, ymin), color = vis_cntr_data
        color = list(map(int, color))
        if color == [0, 0, 0] and not show_low_prob:
            continue
        cv.drawContours(image, cntr, -1, color, 2)
        if text is not None:
            cv.putText(image, text,
                       (xmin, ymin - 20), cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # # Find nearest to the camera contour and draw bold blue line
    if show_nearest and not draw_only_nearest:
        cntr = cv.findContours(
            masks[np.argmin(depth_dists)], cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]

        cv.drawContours(image, cntr, -1, (255, 0, 0), 4)
    return image


def get_padded_image(img):
    # Getting the bigger side of the image
    s = max(img.shape[0:2])

    # Creating a dark square with NUMPY
    f = np.zeros((s, s, 3), np.uint8)

    # Getting the centering position
    ax, ay = (s - img.shape[1])//2, (s - img.shape[0])//2

    # Pasting the 'image' in a centering position
    f[ay:img.shape[0]+ay, ax:ax+img.shape[1]] = img

    return f


def get_centers(cntrs):
    if isinstance(cntrs, list):
        centers = []

        for cntr in cntrs:
            M = cv.moments(cntr)
            # assert M['m00'] != 0
            if M['m00'] == 0:
                print(cntr)

            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centers.append([cX, cY])
        return np.array(centers)

    else:
        M = cv.moments(cntrs)
        # assert M['m00'] != 0
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        centers = [cX, cY]
        return np.array(centers)


def inpaint_depth(depth, factor=1, kernel_size=3, dilate=False):
    """ inpaint the input depth where the value is equal to zero

    Args:
        depth ([np.uint8]): normalized depth array [H, W, 3] (0 ~ 255)
        factor (int, optional): resize factor in depth inpainting. Defaults to 4.
        kernel_size (int, optional): kernel size in depth inpainting. Defaults to 5.

    Returns:
        [np.uint8]: inpainted depth array [H, W, 3] (0 ~ 255)
    """

    H, W, _ = depth.shape
    resized_depth = cv.resize(depth, (W//factor, H//factor))
    mask = np.all(resized_depth == 0, axis=2).astype(np.uint8)
    if dilate:
        mask = cv.dilate(mask, np.ones(
            (kernel_size, kernel_size), np.uint8), iterations=1)
    inpainted_data = cv.inpaint(
        resized_depth, mask, kernel_size, cv.INPAINT_TELEA)
    inpainted_data = cv.resize(inpainted_data, (W, H))
    depth = np.where(depth == 0, inpainted_data, depth)
    return depth


def get_padded_image(img):
    # Getting the bigger side of the image
    s = max(img.shape[0:2])

    # Creating a dark square with NUMPY
    f = np.zeros((s, s, 3), np.uint8)

    # Getting the centering position
    ax, ay = (s - img.shape[1])//2, (s - img.shape[0])//2

    # Pasting the 'image' in a centering position
    f[ay:img.shape[0]+ay, ax:ax+img.shape[1]] = img

    return f


def get_nearest_to_center_box(im_shape, boxes):
    center = np.array(im_shape[:-1]) // 2
    min_dist = 1000000  # just a big number
    min_idx = -1
    for idx, box in enumerate(boxes):
        box_center = ((box[3] + box[1]) // 2, (box[2] + box[0]) // 2)
        dist = euclidean(box_center, center)
        if dist < min_dist:
            min_dist = dist
            min_idx = idx

    return min_idx


def get_one_mask(boxes, mask, image, n_mask=None):
    if n_mask is None:
        cent_ix = get_nearest_to_center_box(image.shape, boxes)
    else:
        cent_ix = n_mask
    x1, y1, x2, y2 = boxes[cent_ix]
    sz = (x2 - x1, y2 - y1)
    mask_rs = cv.resize(mask[cent_ix].squeeze().detach().cpu().numpy(), sz)

    cur_mask = np.zeros((image.shape[: -1]))
    cur_mask[y1: y2, x1: x2] = mask_rs
    ret, res = cv.threshold(cur_mask, 0.5, 1.0, cv.THRESH_BINARY)

    return res


def removeOutliers(x, outlierConstant):
    cur_x = x.clone()
    for col in range(x.shape[1]):
        a = cur_x[:, col]

        upper_quartile = np.percentile(a, 75)
        lower_quartile = np.percentile(a, 25)
        IQR = (upper_quartile - lower_quartile) * outlierConstant
        quartileSet = (lower_quartile - IQR, upper_quartile + IQR)

        cur_x = cur_x[np.where((a >= quartileSet[0]) & (a <= quartileSet[1]))]
        print(cur_x.shape)

    # print(cur_x)
    return cur_x
    # return np.where((a >= quartileSet[0]) & (a <= quartileSet[1]))


def save_data(self, images_masked, im_shape, boxes):

    center_idx = get_nearest_to_center_box(
        im_shape, boxes.cpu().numpy())

    # augment masked images before passing to embedder
    imgs = [self.transforms(image=images_masked[center_idx])[
        'image'] for _ in range(5)]

    features = self.embedder(imgs)

    # check if bounding box is very different from previous

    if self.check_sim(boxes[center_idx].cpu().numpy()):
        if self.x_data_to_save is None:
            self.x_data_to_save = features.squeeze()
        else:
            self.x_data_to_save = torch.cat(
                [self.x_data_to_save, features.squeeze()])
    return center_idx


# taken from https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
def bb_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def get_nearest_mask_id(depth, masks):

    if len(masks) == 0:
        return None

    depth_dists = []
    for mask in masks.astype(np.uint8):

        contours, _ = cv.findContours(
            mask, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            continue

        M = cv.moments(contours[0])
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        if depth[cY, cX] != 0:
            depth_dists.append(depth[cY, cX])
        else:
            depth_dists.append(1e4)
    return np.argmin(depth_dists)
