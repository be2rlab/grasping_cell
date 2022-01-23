import faiss
import pandas as pd

import numpy as np
import torch
import os
from scipy import stats as s
from scipy.spatial.distance import cdist


class knn:
    def __init__(self, knn_file=None, savefile=None, knn_size=10, save_to_file=True, pca_ckpt=None, **kwargs):

        self.knn_size = knn_size
        self.x_data = None
        self.y_data = None
        self.save_file = knn_file if not savefile else savefile
        self.classes = []

        self.save_to_file = save_to_file

        self.faiss_index = None
        # self.faiss_index = faiss.IndexFlatL2()

        self.pca = torch.load(pca_ckpt) if pca_ckpt is not None else None

        if knn_file:
            print(f'loading data from file: {knn_file}')
            if (os.path.exists(knn_file)):
                print('File found')
                data = torch.load(knn_file)
                if not isinstance(data['x'], np.ndarray):
                    data['x'] = np.array(data['x'])
                self.x_data = data['x']

                if self.pca is not None:
                    self.x_data = self.pca.transform(
                        self.x_data).astype(np.float32)
                self.y_data = data['y']
                print(
                    f'Found {self.x_data.shape} points with {len(set(self.y_data))} classes')
                print(pd.Series(self.y_data).value_counts())
                self.classes = list(set(self.y_data))

                # if torch.cuda.is_available():
                #     self.x_data = self.x_data.cuda()

                # self.outlier_estimator.fit(self.x_data.cpu())
                self.faiss_index = faiss.IndexFlatL2(self.x_data.shape[-1])

                self.faiss_index.add(self.x_data)
                # print(self.x_data.shape)
            else:
                print('File not found')

    def print_info(self):
        # print(
        #     f'Found {self.x_data.shape[0]} points with {len(set(self.y_data))} classes')

        print(pd.Series(self.y_data).value_counts())

    def add_points(self, x, y):

        # print(x.shape, len(y))
        if self.x_data is None:
            self.x_data = np.array(x)
            if self.pca is not None:
                self.x_data = self.pca.transform(
                    self.x_data).astype(np.float32)
            self.y_data = y
            self.faiss_index = faiss.IndexFlatL2(self.x_data.shape[-1])
            # print(self.x_data.shape)
        else:
            # self.x_data = torch.cat([self.x_data, x])

            if self.pca is not None:
                x = self.pca.transform(
                    x).astype(np.float32)

            self.x_data = np.concatenate([self.x_data, x])
            self.y_data = self.y_data + y

            self.faiss_index.reset()
            self.faiss_index.add(self.x_data)

        self.classes = list(set(self.y_data))

        # print(self.faiss_index.ntotal)

        if self.save_to_file:
            torch.save({'x': self.x_data,
                        'y': self.y_data}, self.save_file)

    def remove_class(self, cl):
        inds_to_keep = [idx for idx, el in enumerate(self.y_data) if el != cl]

        self.x_data = self.x_data[inds_to_keep]

        if self.pca is not None:
            self.x_data = self.pca.transform(self.x_data).astype(np.float32)

        self.y_data = [self.y_data[i] for i in inds_to_keep]

        self.classes = list(set(self.y_data))

        self.faiss_index.add(self.x_data)
        if self.save_to_file:
            torch.save({'x': self.x_data,
                        'y': self.y_data}, self.save_file)

    def classify(self, x):
        if self.x_data is None:
            # print('No trained data found')
            return None, None, None
        if not isinstance(x, np.ndarray):
            x = np.array(x)

        if self.pca is not None:
            x = self.pca.transform(x).astype(np.float32)
        D, I = self.faiss_index.search(x, self.knn_size)

        D = np.sqrt(D)

        near_y = np.vectorize(lambda a: self.y_data[a])(I)

        cl = s.mode(near_y.T)[0][0]

        frac = [np.count_nonzero(y == row) /
                self.knn_size for y, row in zip(near_y, cl)]

        return cl, frac, D[..., 0]


if __name__ == '__main__':
    knn = knn('/home/iiwa/ros_ws/src/grasping_vision/scripts/datafiles/test_data_own_21_09_dino.pth',
              save_to_file=False)

    # knn.add_points()

    knn.print_info()

    print(knn.faiss_index.is_trained)
    print(knn.faiss_index.ntotal)

    test_data = np.ones((5, 384), dtype=np.float32)

    ret = knn.classify(test_data)

    print(ret)
