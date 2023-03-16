# ---
# jupyter:
#   jupytext:
#     formats: notebooks///ipynb,notebooks/py///py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python [conda env:pytorch]
#     language: python
#     name: pytorch
# ---

# +
import time
import os
from tqdm import tqdm

import pandas as pd
import numpy as np
import cv2
import seaborn as sns
from PIL import Image

import matplotlib.pyplot as plt
# %matplotlib inline
# -

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device', device)

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import os

train_dataset_csv = pd.read_csv('Train_Data.csv')
# cv_dataset = pd.read_csv('CV_Data.csv')
# test_dataset = pd.read_csv('Test_Data.csv')

print(type(train_dataset_csv))
print(len(train_dataset_csv))
train_dataset_csv.head()

pd.read_csv('Train_Data.csv')


# +
class DatasetFromCSV(Dataset):
    def __init__(self, csv_path):
        # 读取csv文件
        self.data_info = pd.read_csv(csv_path)# 去掉header=None，会读取表头
        #第一列为person_id，以下都为一个列表
        self.person_id = np.asarray(self.data_info.iloc[:, 0])
        self.image1 = np.asarray(self.data_info.iloc[:,1])
        self.image2 = np.asarray(self.data_info.iloc[:,2])
        self.report = np.asarray(self.data_info.iloc[:,3])
        # 长度，train=2758
        self.data_len = len(self.data_info.index)

    def __getitem__(self, index):
        singel_person_id=self.person_id[index]
        singel_image1=Image.open(self.image1[index])
        singel_image2=Image.open(self.image2[index])
        singel_report=self.report[index]
        
        transf=transforms.Compose([transforms.Resize((224,224)),
                                   transforms.ToTensor(),
                                  ])
        singel_image1=transf(singel_image1)
        singel_image2=transf(singel_image2)
        ##把图片标签也变成tensor类型
#         singel_report=torch.tensor(singel_report)

#         # Transform image to tensor
#         img_as_tensor = self.to_tensor(img_as_img)
#         # Get label of the image based on the cropped pandas column
#         single_image_label = self.label_arr[index]
#         return singel_person_id,singel_image1
        return [singel_image1,singel_image2], singel_report, singel_person_id

    def __len__(self):
        return self.data_len
# -



train_dataset = DatasetFromCSV('Train_Data.csv')
cv_dataset = DatasetFromCSV('CV_Data.csv')
test_dataset = DatasetFromCSV('Test_Data.csv')

train_dataset[1]

train_loader=DataLoader(dataset=train_dataset,
                        batch_size=16,
                        shuffle=True,
                        drop_last=True,#!!!!!因为Expected input batch_size (4) to match target batch_size (16).错误的原因，暂时修改为TRUE
                        num_workers=0)

test_loader=DataLoader(dataset=test_dataset,
                        batch_size=64,
                        shuffle=True,
                        drop_last=True,
                        num_workers=0)
cv_loader=DataLoader(dataset=cv_dataset,
                        batch_size=64,
                        shuffle=True,
                        drop_last=True,
                        num_workers=0)

# +
import clip

clip.available_models()

# +
model, preprocess = clip.load("ViT-B/32",device=device,jit=False,)
# model.cuda().eval()
input_resolution = model.visual.input_resolution
context_length = model.context_length
vocab_size = model.vocab_size

print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
print("Input resolution:", input_resolution)
print("Context length:", context_length)
print("Vocab size:", vocab_size)
# -

preprocess

model

# +
epoch = 15
batch_size = 16
learning_rate = 5e-5

loss_img = nn.CrossEntropyLoss().to(device)
loss_txt = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)


# -


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()


for i in range(epoch):
    print('----[%d] epoch---' % (i + 1))
    for batch in train_loader:

        images, report, person_id = batch
        report = clip.tokenize(texts=report, truncate=True).to(device)#truncate=True截断句子
        # images = images.to(device)
        image1 = images[0].to(device)
        image2 = images[1].to(device)

        # 通过logits_per_image, logits_per_text = model(images, texts)可以得到预测结果，与torch.arange(N)计算交叉熵进行优化
        logits_per_image1, logits_per_text = model(image1, report)
        if device == "cpu":
            ground_truth = torch.arange(batch_size).long().to(device)
            print("cpu")
        else:
            ground_truth = torch.arange(batch_size, dtype=torch.long, device=device)

        # 反向传播
        total_loss = (loss_img(logits_per_image1, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
        optimizer.zero_grad()
        total_loss.backward()
        if device == "cpu":
            optimizer.step()
        else:
            convert_models_to_fp32(model)
            optimizer.step()
            clip.model.convert_weights(model)

    print('[%d] loss: %.6f' % (i + 1, total_loss))
torch.save(model, 'D:/coding/Jupyter/CLIP-Medical Report Generation/notebooks/model1.pkl')
for i in range(epoch):
    print('----[%d] epoch---' % (i + 1))
    for batch in train_loader:

        images, report, person_id = batch
        report = clip.tokenize(texts=report, truncate=True).to(device)#truncate=True截断句子
        # images = images.to(device)
        image1 = images[0].to(device)
        image2 = images[1].to(device)

        # 通过logits_per_image, logits_per_text = model(images, texts)可以得到预测结果，与torch.arange(N)计算交叉熵进行优化
        logits_per_image1, logits_per_text = model(image1, report)
        logits_per_image2, logits_per_text = model(image2, report)
        logits_per_image = (logits_per_image1+logits_per_image2)/2
        if device == "cpu":
            ground_truth = torch.arange(batch_size).long().to(device)
            print("cpu")
        else:
            ground_truth = torch.arange(batch_size, dtype=torch.long, device=device)

        # 反向传播
        total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
        optimizer.zero_grad()
        total_loss.backward()
        if device == "cpu":
            optimizer.step()
        else:
            convert_models_to_fp32(model)
            optimizer.step()
            clip.model.convert_weights(model)

    print('[%d] loss: %.6f' % (i + 1, total_loss))
torch.save(model, 'D:/coding/Jupyter/CLIP-Medical Report Generation/notebooks/model2.pkl')


if hasattr(torch.cuda, 'empty_cache'):
	torch.cuda.empty_cache()


