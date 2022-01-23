import cv2
from uoais.uoais_model import uoais_model
import json

from faiss_knn.faiss_knn import knn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import os
import cv2 as cv
import torch
import random
import time
import numpy as np
import pandas as pd
import glob

from models.all_model import AllModel
from matplotlib import pyplot as plt
import seaborn as sns


def get_iou(mask, gt_regions):

    pred_mask = np.zeros(mask.shape)
    pred_mask[mask] = 255
    # cv.imshow('mask', pred_mask)

    ious = []
    clss = []
    for r in zip(*gt_regions):
        gt_mask = np.zeros(r[0].shape)
        gt_mask[r[0]] = 255

        intersection = cv.bitwise_and(pred_mask, gt_mask)
        union = cv.bitwise_or(pred_mask, gt_mask)

        iou = (intersection != 0).sum() / (union != 0).sum()
        ious.append(iou)
        clss.append(r[1])

        # cv.imshow('gt', np.hstack([pred_mask, gt_mask, intersection, union]))
        # cv.waitKey()
    if len(ious) > 0:
        max_ids = np.argmax(ious)
        iou = ious[max_ids]
        cl = clss[max_ids]
    else:
        iou = 0.0
        cl = 0
        max_ids = -1
    return iou, cl, max_ids


def evaluate_predictions(pred_regions, gt_regions):
    # (insts['pred_masks'], clss), gt_regions)

    ious = []
    gt_clss = []
    gt_idxs = []
    clss = pred_regions[1]
    confs = pred_regions[2]

    for mask in pred_regions[0]:
        iou, gt_cl, gt_idx = get_iou(mask, gt_regions)
        ious.append(iou)
        gt_clss.append(gt_cl)
        gt_idxs.append(gt_idx)

    df = pd.DataFrame({'pred_cl': clss, 'conf': confs,
                      'gt_cl': gt_clss, 'gt_idx': gt_idxs, 'iou': ious})

    return df


root = 'All8'
# root = 'dataset_for_test'

color_files = glob.glob(f'{root}/*rgb*.png')
# print(color_files)

torch.set_grad_enabled(False)

model = AllModel(knn_file='knn_data_vits16.pth',
                 segm_conf_thresh=0.5,
                 segm_nms_thresh=0.5,
                 preprocess_depth=False)

with open(f'{root}/via_region_data.json', 'r') as f:
    annots = json.load(f)


# cv.namedWindow('segm+clsf', cv.WINDOW_GUI_NORMAL)
dfs = []
total_objects = 0
for f in tqdm(color_files):
    annot_i = [i for i in annots.keys() if i.startswith(f.split('/')[-1])]

    annot = annots[annot_i[0]]

    color_im = cv.imread(f)

    depth_f = f.replace('rgb', 'depth.cm.8')
    # depth_f = f.replace('rgb', 'depth')
    depth_im = cv.imread(depth_f, -1)
    depth_im = np.uint32(depth_im) * 10

    depth_im = np.zeros_like(depth_im)

    start = time.time()

    # (clss, confs, dists), insts, vis_masks = model(color_im, depth_im[:, :, 0])
    (clss, confs, dists), insts, vis_masks = model(color_im, depth_im)

    show_vis_masks = vis_masks.copy()

    # stacked_imgs = []
    for cls, conf, d, box in zip(clss, confs, dists, insts.pred_boxes.tensor):
        x, y, w, h = list(map(int, box))
        cv.putText(vis_masks, f'{cls} {conf:.2f} {d:.2f}', (x - 2,
                                                            y - 2), cv.FONT_HERSHEY_PLAIN, 1.1, (0, 0, 255))

    # convert annotations to polygons
    gt_regions = []

    total_objects += len(annot['regions'])

    for region in annot['regions']:
        ptsx = region['shape_attributes']['all_points_x']
        ptsy = region['shape_attributes']['all_points_y']

        label = region['region_attributes']['name']

        pts = np.array([ptsx, ptsy], dtype=np.int32).T
        pts = np.expand_dims(pts, 1)

        mask = np.zeros((480, 640))
        cv.fillPoly(mask, [pts], True, 255)

        cv.polylines(vis_masks, [pts], True, (0, 0, 255))

        mask = mask != 0

        gt_regions.append((mask, label))
    gt_regions = list(map(list, zip(*gt_regions)))
    pred_masks = insts.pred_masks
    df = evaluate_predictions((pred_masks, clss, insts.scores), gt_regions)
    df['filename'] = f
    dfs.append(df)
    cv2.putText(vis_masks, f, (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)
    # cv.imshow('segm+clsf', vis_masks)

    # cv.imwrite(f'output/{f.split("/")[-1]}', vis_masks)
    key = cv.waitKey(0)
    # print(key, ord('q'))
    # if key == ord('q') & 0xFF:
    #     cv.destroyAllWindows()
    #     exit()

# print(total_objects)
df = pd.concat(dfs, axis=0, ignore_index=True)
df_sorted = df.sort_values('iou', ascending=False)
df_wo_dupls = df_sorted.drop_duplicates(['gt_idx', 'filename'], keep='first')
first_ids = df_wo_dupls.index.values.tolist()
df_sorted['first_tp'] = df_sorted.index
df_sorted['first_tp'] = df_sorted['first_tp'].apply(lambda x: x in first_ids)

df_sorted = df_sorted.sort_values('conf', ascending=False)


# print(df)
df_sorted.to_csv(f'test_results_{root}.csv', index_label=False)

# calculate metrics on predictions
mAPs = []
for iou_thresh in np.linspace(0.5, 0.95, 10):
    df = pd.read_csv(f'test_results_{root}.csv')

    def func(x): return x[0] > (iou_thresh) & x[1]

    df['TP/FP'] = np.logical_and(df['iou']
                                 > iou_thresh, df['first_tp'])

    df.loc[df.iou == 0, 'gt_cl'] = np.nan

    acc_tp = np.ones(len(df), dtype=np.int32) * df['first_tp'][0].astype(int)
    acc_fp = np.ones(len(df), dtype=np.int32) * (not df['first_tp'][0])

    tf_array = df['TP/FP'].to_numpy()

    for i in range(1, len(acc_tp)):

        acc_tp[i] = acc_tp[i-1] + tf_array[i]
        nT = tf_array[i]
        acc_fp[i] = acc_fp[i-1] + (not tf_array[i])

    df['acc_tp'] = acc_tp
    df['acc_fp'] = acc_fp

    df['precision'] = df['acc_tp'] / (df['acc_tp'] + df['acc_fp'])

    total = len(df.dropna()['gt_cl'])
    df['recall'] = df['acc_tp'] / total_objects

    max_precisions = []
    for cur_recall in np.linspace(0, 1, 11):
        min_v = min(df['recall'], key=lambda x: abs(x-cur_recall))

        max_precision = df[df['recall'] > min_v]['precision'].max()
        if max_precision is np.nan:
            max_precision = 0
        max_precisions.append(max_precision)

    # sns.lineplot(df['recall'], df['precision'], markers='*')
    # plt.show()
    AP = np.mean(max_precisions)
    print(f'AP at IOU {iou_thresh:.2f}: {AP:.2f}')

    mAPs.append(AP)
# print(df.head(30))

print(f'mAP COCO: {np.mean(mAPs):.2f}')
