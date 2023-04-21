import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
import argparse
import os
import copy
from torchvision import models
from torchsummary import summary
import time
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras import backend as K
from keras.layers import Conv2D,MaxPooling2D,UpSampling2D,Input,BatchNormalization,LeakyReLU
from keras.layers import concatenate
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import Project

LABELS_Severity = {35: 0,
                   43: 0,
                   47: 1,
                   53: 1,
                   61: 2,
                   65: 2,
                   71: 2,
                   85: 2}


mean = (.1706)
std = (.2112)
normalize = transforms.Normalize(mean=mean, std=std)

transform = transforms.Compose([
    transforms.Resize(size=(224,224)),
    transforms.ToTensor(),
    normalize,
])

if __name__ == '__main__':
    args = Project.parse_args()
    
    trainset = Project.OCTDataset(args, 'train', transform=transform)
    testset = Project.OCTDataset(args, 'test', transform=transform)

    train_dataloader = DataLoader(trainset, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(testset, batch_size=1, shuffle=True)

    model = torch.jit.load('AlexNet_normal.pt')
    model.eval()
    appeared = [0, 0, 0]
    correct = [0, 0, 0]
    pred = [0,0,0]
    
    with torch.no_grad():
        for i in range(7987):
            img, target, eye = next(iter(test_dataloader)) # extract image from train loader
            img = img.cuda()
            target = target.cuda().item()
            out = model(img)# extract output vector using model
                #print()
            pred_label = out.argmax().item() #LABELS_Severity[labels[out.argmax().item()]] # obtain predicted label
            if pred_label == target:
                correct[pred_label]+=1
                #print("preded: ", pred_label)
                #print("actual: ", target)
            pred[pred_label]+=1
            appeared[target] += 1
    
    print(correct)
    print(pred)
    print(appeared)
