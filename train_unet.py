# -*- coding: utf-8 -*-
__author__ = "Yu-Sheng Lin"
__copyright__ = "Copyright (C) 2016-2021"
__license__ = "AGPL"
__email__ = "pyquino@gmail.com"

from torchsummary import summary
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from torch import  optim
from core.unet import UNet
import torchvision.transforms as T

import os
import math
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from sementic_seg_dataloader import CLASIFER, showimg, mean_iou_score, TOOLDATA


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def train(model, epoch, log_interval=5):
    optimizer = optim.Adam(model.parameters(), 1e-5)
    criterion = nn.CrossEntropyLoss()
    model.train()  # Important: set training mode
    iteration = 0
    for ep in range(epoch):
        for batch_idx, (data, target) in enumerate(trainloader):
            data, target = data.to(device), target.to(device, dtype=torch.long)

            output = model(data)
            # print(f"output shape {output.shape} and input shape{data.shape} and target shape {target.shape}")
            optimizer.zero_grad()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if iteration % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    ep, batch_idx * len(data), len(trainloader.dataset),
                    100. * batch_idx / len(trainloader), loss.item()))
            iteration += 1

        accuracy = test(model, valsetloader)  # Evaluate at the end of each epoch
        acc_train = test(model, trainloader)
        print(f"accuracy of train data {acc_train}, acc for val {accuracy}")
        if accuracy > 0.66:
            torch.save(model.state_dict(), f"save_model/{ep}_{accuracy:.2f}.pt")


def test(model, valsetloader):
    criterion = nn.CrossEntropyLoss()
    model.eval()  # Important: set evaluation mode
    test_loss = 0
    correct = 0
    with torch.no_grad():  # This will free the GPU memory used for back-prop
        iou_outputs = []
        iou_labels = []
        for data, target in valsetloader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            output = torch.argmax(output, dim=1)
            for i in range(output.shape[0]):
                out = output[i]
                tar = target[i]
                iou_outputs.append(out.cpu().detach().numpy())
                iou_labels.append(tar.cpu().detach().numpy())
        
        meaniou = mean_iou_score(np.concatenate(iou_outputs), np.concatenate(iou_labels))
        # print(f"mean iou accuracy {t}")
    return meaniou


def pred_one_image(model, valsetloader):



    # print(valset[21])

    model.eval()  # Important: set evaluation mode
    test_loss = 0
    correct = 0
    with torch.no_grad():  # This will free the GPU memory used for back-prop

        iou_outputs = []
        iou_labels = []
        for data, target in valsetloader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            output = torch.argmax(output, dim=1)
            for i in range(output.shape[0]):
                out = output[i]
                tar = target[i]
                iou_outputs.append(out.cpu().detach().numpy())
                iou_labels.append(tar.cpu().detach().numpy())
        exit()
        t = 0
        meaniou = mean_iou_score(np.concatenate(iou_outputs), np.concatenate(iou_labels))
        print(f"mean iou accuracy {meaniou}")
    return t


if __name__ == '__main__':

    trainset = TOOLDATA(root="train_data/image/imm", val_or_test=False, transform=transforms.ToTensor())
    valset = TOOLDATA(root="val_data/imm", val_or_test=False, transform=transforms.ToTensor())
    
    # model = UNet(n_channels=1, n_classes=2)
    # model.load_state_dict(torch.load("save_model/2_0.93.pt"))
    # model.eval()
    # with torch.no_grad():
        # transforms = T.Compose([T.ToTensor()])
        # data, label = valset[0]
        # print(data)
        # output = model(data.unsqueeze(0))
        # output = torch.argmax(output, dim=1)
    # print(output.size())
    # imgg = showimg(output.permute(1, 2, 0))
   
    # plt.imshow(imgg)
    # plt.show()
    # exit()
    # print(trainset[0])
    
    # print(trainset[0][1])
    # exit()

    trainloader = DataLoader(trainset, batch_size=8, shuffle=True, num_workers=4)
    valsetloader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=1)

    # model = UNet(n_channels=1, n_classes=2)
    # model.load_state_dict(torch.load("save_model/85_0.68.pt"))
    # model.to(device)
    # pred_one_image(model, valsetloader)
    # exit()

    model = UNet(n_channels=1, n_classes=2)
    model.to(device)   
    model.to(device)
    train(model, 90)
    torch.save(model.state_dict(), "save_model/unet_t.pt")