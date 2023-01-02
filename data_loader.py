import random

import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
import torch

import os
import numpy as np
import cv2
from PIL import Image

class Chest_Dataset(Dataset):
    def __init__(self, root_dir, train_val_test, transform=None):
        self.root_dir = root_dir
        self.test = True if train_val_test == "test/" else False
        self.dataset_dir_name = os.path.join(root_dir, train_val_test) #chest_xray/train/
        self.transform = transform
        self.labels = ['NORMAL', 'PNEUMONIA']
        self.images_with_labels = self._get_images_with_labels()

    def __getitem__(self, index):
        if self.test == False:
            image_arr, image_label = self.images_with_labels[index]
            image_data = Image.fromarray(image_arr)
            transform_arr = self.transform(image_data)# / np.max(image_data)
            return transform_arr, image_label
        else: #testing
            image_arr, img_idx = self.images_with_labels[index]
            image_data = Image.fromarray(image_arr)
            transform_arr = self.transform(image_data)# / np.max(image_data)
            return transform_arr, img_idx

    def __len__(self):
        if self.test == False:
            num = 0
            for label in self.labels:
                path = os.path.join(self.dataset_dir_name, label)
                num = num + len(os.listdir(path))
            return num
        else: # testing
            return len(os.listdir(self.dataset_dir_name))

    def _get_images_with_labels(self):
        if self.test == False:
            images = list()
            for label in self.labels: 
                path = os.path.join(self.dataset_dir_name, label)
                class_num = self.labels.index(label)
                for img in os.listdir(path):
                    try:
                        img_arr = cv2.imread(os.path.join(path, img))
                        image_arr = cv2.cvtColor(img_arr,cv2.COLOR_BGR2RGB)
                        images.append([image_arr, class_num])
                    except Exception as e:
                        print(e)
            return images
        else: #testing
            images = list()
            path = self.dataset_dir_name
            for img in os.listdir(path):
                try:
                    img_arr = cv2.imread(os.path.join(path, img))
                    image_arr = cv2.cvtColor(img_arr,cv2.COLOR_BGR2RGB)
                    images.append([image_arr, img])
                except Exception as e:
                    print(e)
            return images

def get_numpy_data2d_with_labels(data_loader):
    data2d = None
    labels = list()
    for data, target in data_loader:
        #print(data.shape)
        batch_data2d = data.numpy()[:, 0, :, :].reshape(data.shape[0], -1)
        data2d = batch_data2d if data2d is None else np.vstack((data2d, batch_data2d))
        #print(data2d.shape)
        labels.extend(target.numpy())
    labels = np.array(labels)
    #print(data2d.shape)
    #print(labels.shape)
    return data2d, labels

def get_numpy_data2d(data_loader):
    data2d = None
    idxs = list()
    for data, img_idx in data_loader:
        batch_data2d = data.numpy()[:, 0, :, :].reshape(data.shape[0], -1)
        data2d = batch_data2d if data2d is None else np.vstack((data2d, batch_data2d))
        idxs.extend(img_idx)
    idxs = np.array(idxs)
    #print(idxs)
    return data2d, idxs