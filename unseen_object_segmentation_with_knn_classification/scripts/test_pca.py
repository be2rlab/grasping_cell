from torch._C import dtype
from OCID_dataset import OCID_classification_dataset
import cv2 as cv
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import seaborn as sns
from umap import UMAP
from matplotlib import pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA


d = torch.load('features_w_labels_vits16.pth')
x_data, y_data = d['x'], d['y']

print(x_data.shape)

# pca = PCA(n_components=x_data.shape[1])

# x_data_red = pca.fit_transform(x_data)


x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.3, random_state=42, shuffle=True)

# print('shapes of arrays:', x_train.shape, len(y_train))
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(x_train, y_train)

preds = knn.predict(x_test)
M = confusion_matrix(y_test, preds)
# print(M)
accs = M.diagonal()/M.sum(axis=1)
print('wo PCA, accuracy: ', accs.mean())

print('----------------------------------------------------------')

# fig = plt.figure()
# ax = fig.add_subplot(2, 1, 1)

# ax.plot(pca.explained_variance_ratio_[:750])

# ax.set_yscale('log')
# plt.show()
fs = os.listdir('pca')
fs.sort(key=lambda x: int(x.split('_')[2]))
for f in fs:
    pca = torch.load(f'pca/{f}')
    x_data_red = pca.transform(x_data)

    x_train, x_test, y_train, y_test = train_test_split(
        x_data_red, y_data, test_size=0.3, random_state=42, shuffle=True)

    # print('shapes of arrays:', x_train.shape, len(y_train))
    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(x_train, y_train)

    preds = knn.predict(x_test)
    M = confusion_matrix(y_test, preds)
    # print(M)
    accs = M.diagonal()/M.sum(axis=1)
    print(f, 'PCA, accuracy: ', accs.mean())

    print('----------------------------------------------------------')
