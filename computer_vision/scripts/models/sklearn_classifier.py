import os

import pandas as pd
import numpy as np
import torch

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import LocalOutlierFactor


class SKLearnClassifierWrapper:
    def __init__(self, knn_file=None, savefile=None, save_to_file=True, model=None, **kwargs):

        self.x_data = None
        self.y_data = None
        self.save_file = knn_file if not savefile else savefile
        self.classes = []

        self.save_to_file = save_to_file
        if not model:
            self.model = KNeighborsClassifier(
                n_neighbors=10, weights='distance', metric='cosine', n_jobs=-1)
        else:
            self.model = model
        self.le = LabelEncoder()

        # self.outlier_detector = SGDOneClassSVM(tol=1e-6)
        self.outlier_detector = LocalOutlierFactor(
            novelty=True, metric='cosine', n_neighbors=5)

        if knn_file:
            print(f'loading data from file: {knn_file}')
            if (os.path.exists(knn_file)):
                print('File found')
                data = torch.load(knn_file)
                self.add_points(data['x'], data['y'])

                print(
                    f'Found {self.x_data.shape[0]} points with {len(set(self.y_data))} classes')
                print(pd.Series(self.y_data).value_counts())

            else:
                print('File not found')

    def print_info(self):
        print(pd.Series(self.y_data).value_counts())

    def add_points(self, x, y):
        if self.x_data is None:
            self.x_data = np.array(x)
            self.y_data = y
        else:
            self.x_data = np.concatenate([self.x_data, x])
            self.y_data = self.y_data + y

        self.classes = list(set(self.y_data))
        self.label_data = self.le.fit_transform(self.y_data)
        self.outlier_detector.fit(self.x_data)
        print(self.outlier_detector.offset_)
        self.model.fit(self.x_data, self.label_data)

        if self.save_to_file:
            torch.save({'x': self.x_data,
                        'y': self.y_data}, self.save_file)

    def remove_class(self, cl):
        inds_to_keep = [idx for idx, el in enumerate(self.y_data) if el != cl]

        self.x_data = self.x_data[inds_to_keep]
        self.y_data = [self.y_data[i] for i in inds_to_keep]

        self.classes = list(set(self.y_data))

        self.outlier_detector.fit(self.x_data)
        self.model.fit(self.x_data, self.y_data)
        if self.save_to_file:
            torch.save({'x': self.x_data,
                        'y': self.y_data}, self.save_file)

    def classify(self, x):
        if self.x_data is None:
            return None, None, None
        if not isinstance(x, np.ndarray):
            x = np.array(x)

        probs = self.model.predict_proba(x)
        max_ids = np.argmax(probs, axis=1)

        classes = self.le.inverse_transform(max_ids)
        confs = np.max(probs, axis=1)

        outliers = self.outlier_detector.predict(x)
        scores = self.outlier_detector.decision_function(x)

        D = np.array([1000.0 if o == -1 else 0.0 for o in outliers])
        D += scores

        return classes, confs, D
