# -*- coding: utf-8 -*-
__author__ = "Yu-Sheng Lin"
__copyright__ = "Copyright (C) 2016-2021"
__license__ = "AGPL"
__email__ = "pyquino@gmail.com"

import torch
import os
import math
import torch.nn.functional as F
import glob
import numpy as np
import random
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import platform

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


# colormap for mask data
# colormap = np.array([(0, 0, 0),  # 0=Unknow
#                      (255, 255, 255),  # 1=Barren,
#                      (0, 0, 255),  # 2=Water
#                      (0, 255, 0),  # 3=Forest
#                      (255, 0, 255),  # 4=Rangeland
#                      (255, 255, 0),  # 5=Agriculture
#                      (0, 255, 255),  # 6=Urban
#                      ])

colormap = np.array([(0, 0, 0),  # 0=Unknow
                     (255, 255, 255),  # 1=Barren,
                    ])
# classes = ['Unknow', 'Barren', 'Water', 'Forest', 'Rangeland', 'Agriculture', 'Urban']
classes = ["BackGround", "Tool Flank"]
num_classes = len(classes)

cm2lbl = np.zeros(256 ** 3)
for i, cm in enumerate(colormap):
    """Build the index of the color map"""
    cm2lbl[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i  # 建立索引

def image2label(im):
    """find the mast index of map"""
    data = np.array(im, dtype='int32')
    idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
    return np.array(cm2lbl[idx], dtype='int64')  # 根據索引得到 label 矩陣
    
def image2label_test(im):
    """find the mast index of map"""
    data = np.array(im, dtype='int32')
    return data
    # return np.array(cm2lbl[idx], dtype='int64')  # 根據索引得到 label 矩陣

def showimg(data):  # data shape (512, 512)
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    for i, color in enumerate(colormap):
        indices = np.where(data == i)
        if indices[0].size == 0:
            continue
        coords = zip(indices[0], indices[1])
        for cod in coords:
            img[cod[0], cod[1]] = color
    return img

def mean_iou_score(pred, labels):
    '''
    Compute mean IoU score over 6 classes
    '''
    mean_iou = 0
    for i in range(2):
        tp_fp = np.sum(pred == i)
        tp_fn = np.sum(labels == i)
        tp = np.sum((pred == i) * (labels == i))
        iou = tp / (tp_fp + tp_fn - tp)
        mean_iou += iou / 2.0
        # print('class #%d : %1.5f'%(i, iou))
    # print('\nmean_iou: %f\n' % mean_iou)

    return mean_iou


class CLASIFER(Dataset):
    """Training data for the read image and label"""

    def __init__(self, root, val_or_test: bool, transform=None):
        """Initialize the image dataset"""
        self.images = None
        self.labels = None
        self.filenames = []
        self.root = root
        self.transform = transform
        self.val_or_test = val_or_test
        self.fname = None

        filenames = sorted(glob.glob(os.path.join(self.root, '*.jpg')))
        self.fname = filenames
        filenames_label = sorted(glob.glob(os.path.join(self.root, '*.png')))
        for i in range(len(filenames)):
            self.filenames.append((filenames[i], filenames[i]))  # (filename, label) pair

        self.len = len(self.filenames)


    def __getitem__(self, index):
        """Get a sample from the dataset"""
        image = Image.open(self.filenames[index][0])
        label = Image.open(self.filenames[index][1])

        if self.transform is not None:
            image, label = self.image_transform(image, label)
        # label = image2label(mask)

        return image, label

    def __len__(self):
        """total number of samples in the dataset"""
        return self.len

    def image_transform(self, image, mask):
        """Random transform the image data"""

        # Random crop
        # i, j, h, w = transforms.RandomCrop.get_params(
        # image, output_size=(512, 512))
        # image = TF.crop(image, i, j, h, w)
        # mask = TF.crop(mask, i, j, h, w)

        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        # Random rotate
        angle = transforms.RandomRotation.get_params((0, 360))
        image = TF.rotate(image, angle)
        mask = TF.rotate(mask, angle)

        # Transform to tensor
        image = TF.to_tensor(image)
        # norm_mean = (0.5, 0.5, 0.5)
        # norm_std = (0.5, 0.5, 0.5)
        # image = TF.normalize(image, norm_mean, norm_std)

        return image, mask

class TOOLDATA(Dataset):
    """Training data for the read image and label"""

    def __init__(self, root, val_or_test: bool, transform=None):
        """Initialize the image dataset"""
        self.images = None
        self.labels = None
        self.filenames = []
        self.root = root
        self.transform = transform
        self.val_or_test = val_or_test
        self.fname = None

        filenames = sorted(glob.glob(os.path.join(self.root, '*.jpg')))
        self.fname = filenames
        filenames_label = sorted(glob.glob(os.path.join(self.root, '*.png')))
        
        for i in range(len(filenames)):
            self.filenames.append((filenames[i], filenames_label[i]))  # (filename, label) pair

        self.len = len(self.filenames)

    def __getitem__(self, index):
            """Get a sample from the dataset"""
            image = Image.open(self.filenames[index][0])
            label = Image.open(self.filenames[index][1])
            image = image.resize((512, 512))
            label = label.resize((512, 512))

            if self.transform is not None:
                image, label = self.image_transform(image, label)
            label = image2label_test(label)
            # label = image2label(label)

            return image, label

    def __len__(self):
        """total number of samples in the dataset"""
        return self.len

    def image_transform(self, image, mask):
        """Random transform the image data"""

        # Random crop
        # i, j, h, w = transforms.RandomCrop.get_params(
        # image, output_size=(512, 512))
        # image = TF.crop(image, i, j, h, w)
        # mask = TF.crop(mask, i, j, h, w)

        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random vertical flipping
        # if random.random() > 0.5:
        #     image = TF.vflip(image)
        #     mask = TF.vflip(mask)

        # Random rotate
        angle = transforms.RandomRotation.get_params((0, 360))
        image = TF.rotate(image, angle)
        mask = TF.rotate(mask, angle)

        # Transform to tensor
        image = TF.to_tensor(image)
        # mask = TF.to_tensor(mask)
        # norm_mean = (0.5, 0.5, 0.5)
        # norm_std = (0.5, 0.5, 0.5)
        # image = TF.normalize(image, norm_mean, norm_std)

        return image, mask



class Validation(Dataset):
    """This is for output the data and test data only read the image without label"""

    def __init__(self, root, val_or_test: bool, transform=None):
        """Initialize the image dataset"""
        self.images = None
        self.filenames = []
        self.root = root
        self.transform = transform
        filenames = sorted(glob.glob(os.path.join(self.root, '*.jpg')))
        self.fname = filenames

        for i in range(len(filenames)):
            self.filenames.append((filenames[i]))  # (filename) pair

        self.len = len(self.filenames)

    def __getitem__(self, index):
        """Get a sample from the dataset"""
        image = Image.open(self.filenames[index])

        # This line is for testing data
        image = TF.to_tensor(image)

        return image

    def __len__(self):
        """total number of samples in the dataset"""
        return self.len

    def get_list_filename(self):
        tmp = []
        if platform.system() == "Linux":
            splitter = "/"
        elif platform.system() == "Windows":
            splitter = "\\"
        for f in self.fname:
            name = f.split(splitter)[-1]
            tmp.append(name)
        return tmp