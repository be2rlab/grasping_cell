
import time
import os

import torch
import numpy as np
import cv2 as cv
import albumentations as A
from albumentations.pytorch import ToTensorV2

# local packages
from models.faiss_knn import knn
from models.classifier import classifier
# from models.detectron2_wrapper import Detectron2Wrapper
from models.mmdet_wrapper import MMDetWrapper


from utilities.utils import get_nearest_mask_id


class AllModel:
    # def __init__(self, fe=None, knn_file=None, segm_conf_thresh=None, fe_fp16=False) -> None:
    @torch.no_grad()
    def __init__(self, fe=None, fe_fp16=False, n_augmented_crops=10, dataset_save_folder=f'{os.path.dirname(os.path.realpath(__file__))}../dataset_segmentation', **kwargs) -> None:
        self.fe_fp16 = fe_fp16
        self.n_augmented_crops = n_augmented_crops

        script_dir = os.path.dirname(os.path.realpath(__file__))

        self.dataset_save_folder = dataset_save_folder
        # prepare models for segmentation, feature extraction and classification
        self.segm_model = MMDetWrapper(**kwargs)

        if not fe:
            self.fe = torch.hub.load(
                'facebookresearch/dino:main', 'dino_vits16')
        else:
            self.fe = fe
        self.fe.eval()
        if fe_fp16:
            self.fe.half()

        if torch.cuda.is_available():
            self.fe.cuda()

        self.classifier = knn(**kwargs)
        # self.classifier = classifier(**kwargs)

        # check if saved knn file has same dimensionality as feature extractor
        if self.classifier.x_data is not None:
            sample_input = torch.ones(
                (1, 3, 224, 224), dtype=torch.float32, device='cuda', )
            if fe_fp16:
                sample_input = sample_input.half()


        self.fe_transforms = A.Compose([
            A.LongestMaxSize(max_size=150),
            A.PadIfNeeded(min_height=150, min_width=150, border_mode=cv.BORDER_CONSTANT, p=1),
            ToTensorV2()
        ])
        self.fe_augmentations = A.Compose([
            A.LongestMaxSize(max_size=224),
            A.PadIfNeeded(min_height=224, min_width=224, border_mode=cv.BORDER_CONSTANT, p=1),
            # A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4),
            A.Flip(),
            A.ShiftScaleRotate(p=1),
            A.CenterCrop(150, 150, p=1),
            ToTensorV2()
        ])

        self.features_to_save = []

        self.classes = self.classifier.classes.copy()

    @torch.no_grad()
    def __call__(self, color_im, depth_im, train=False):

        if not train: 
            
            cropped_objs, masks = self.segm_model(color_im)
            if not cropped_objs:
                return (None, None, None), masks

            transformed_objs = torch.stack(
                [self.fe_transforms(image=i)['image'] for i in cropped_objs]).cuda()

            # plt.imshow(cropped_objs[0])
            # plt.show()
            if self.fe_fp16:
                transformed_objs = transformed_objs.half()

            features = self.fe(transformed_objs).cpu().float()

            cls, confs, dists = self.classifier.classify(features)

            return (cls, confs, dists), masks
        else:
            cropped_objs, masks = self.segm_model(color_im)
            if not cropped_objs:
                return False

            nearest_mask_id = get_nearest_mask_id(depth_im, masks)

            # plt.imshow(cropped_objs[nearest_mask_id])
            # plt.show()
            
            if self.n_augmented_crops:
                transformed_objs = [self.fe_augmentations(image=cropped_objs[nearest_mask_id])['image'] for _ in range(self.n_augmented_crops)]
                transformed_objs = torch.stack(transformed_objs).cuda()
            else:
                transformed_objs = torch.tensor(self.fe_transforms(image=cropped_objs[nearest_mask_id])['image']).unsqueeze(0).cuda()

            
            # transformed_objs = torch.stack(
            #     [self.fe_transforms(i) for i in cropped_objs]).cuda()

            if self.fe_fp16:
                transformed_objs = transformed_objs.half()

            features = self.fe(transformed_objs).cpu().float()

            cl = f'{len(self.classifier.classes) + 1}'
            fname = f'{self.dataset_save_folder}/{cl}/{(time.time()):.2f}.png'


            os.makedirs(f'{self.dataset_save_folder}/{cl}', exist_ok=True)
            cv.imwrite(fname, cropped_objs[nearest_mask_id])

            return self.save_feature(features)

            # cls, confs, dists = self.classifier.classify(features)

            # return (masks, features, cropped_objs)# if train else ((cls, confs, dists), masks)


    def save_feature(self, features):
        self.features_to_save += features
        # feats = np.stack(features)
        # self.classifier.add_points(
        #     # feats, [f'object_{len(self.classifier.classes)}'] * len(feats))
        #     feats, [f'{len(self.classifier.classes)}'] * len(feats))
        # self.classifier.print_info()
        # # self.features_to_save = []
        return 'features saved'

    def add_to_knn(self):
        feats = np.stack(self.features_to_save)

        self.classes.append(f'{len(self.classifier.classes) + 1}')

        self.classifier.add_points(
            # feats, [f'object_{len(self.classifier.classes)}'] * len(feats))
            feats, [f'{len(self.classifier.classes) + 1}'] * len(feats))
        self.classifier.print_info()
        self.features_to_save = []

    # def addToKNN(self, masked_objs, labels):

    #     if not isinstance(masked_objs, list):
    #         masked_objs = [masked_objs]

    #     assert len(masked_objs) == len(labels)

    #     transformed_objs = torch.stack(
    #         [self.fe_transforms(image=i)['image'] for i in masked_objs]).cuda()
    #     features = self.fe(transformed_objs).cpu()

    #     self.classifier.add_points(features, labels)
