# -*- coding: utf-8 -*-
__author__ = "Yu-Sheng Lin"
__copyright__ = "Copyright (C) 2016-2021"
__license__ = "AGPL"
__email__ = "pyquino@gmail.com"

# import torch
import os
import math
# import torch.nn.functional as F
import glob
import numpy as np
import random
# import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import platform

# from torch.utils.data import Dataset
# from torchvision import transforms
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

label = Image.open("label1.png").convert('RGB')
print(np.array(label))
mask = image2label(label)
plt.imshow(mask)
plt.show()
print(mask)

def showimg(data):  # data shape (512, 512)
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    for i, color in enumerate(colormap):
        indices = np.where(data == i)
        if indices[0].size == 0:
            continue
        coords = zip(indices[0], indices[1])
        for cod in coords:
            img[cod[0], cod[1]] = color

    # plt.imshow(img)
    # plt.show()
    return img

def mean_iou_score(pred, labels):
    '''
    Compute mean IoU score over 6 classes
    '''
    mean_iou = 0
    for i in range(7):
        tp_fp = np.sum(pred == i)
        tp_fn = np.sum(labels == i)
        tp = np.sum((pred == i) * (labels == i))
        iou = tp / (tp_fp + tp_fn - tp)
        mean_iou += iou / 6
        # print('class #%d : %1.5f'%(i, iou))
    # print('\nmean_iou: %f\n' % mean_iou)

    return mean_iou
