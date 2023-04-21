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
import torch.nn.functional as F
from sklearn.svm import SVC

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
    transforms.Resize(size=(512,512)),
    transforms.ToTensor(),
    normalize,
])

class SuperAE(nn.Module):
    def __init__(self, pretrained_model):
        super().__init__()
        self.__dict__ = pretrained_model.__dict__.copy()
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        # add second hidden layer
        # 256
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # compressed representation
        # 128
        x = F.relu(self.conv3(x))
        x = self.pool(x)  # compressed representation
        return x
if __name__ == '__main__':
    args = Project.parse_args()
    
    trainset = Project.OCTDataset(args, 'train', transform=transform)
    testset = Project.OCTDataset(args, 'test', transform=transform)

    train_dataloader = DataLoader(trainset, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(testset, batch_size=1, shuffle=True)

    model = torch.jit.load('AE_for_visual_final.pt')
        
    #model.eval()
    appeared = [0, 0, 0]
    correct = [0, 0, 0]
    pred = [0,0,0]
    
    with torch.no_grad():
        for i in range(7987):
            img, target, eye = next(iter(test_dataloader)) # extract image from train loader
            img = img.cuda()#.permute(1,0,2,3)
            target = target.cuda().item()
            print(target)
            out, rep = model(img)# extract output vector using model
                #print()
            print(rep.cpu().numpy())
    
    print(correct)
    print(pred)
    print(appeared)
    
    # split the x_train, x_test, y_train, y_test
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    '''for i, (x,y,z) in enumerate(train_dataloader):
        X_train.append(model(x).cpu().numpy().resize(1,-1))
        y_train.append(y)
        print(i,' train')
    for i,(x,y,z) in enumerate(test_dataloader):
        X_test.append(model(x).cpu().numpy().resize(1,-1))
        y_test.append(y)
        print(i,' test')'''

    # initiate svm classifier 
    svmclf = SVC(kernel = 'linear')
    # train the svm model
    svmclf.fit(X_train,y_train)
    y_pred = svmclf.predict(X_test)

    # accuracy score 
    #accuracy = accuracy_score(y_test, y_pred)
    # loss
    #hinge_loss = metrics.hinge_loss(y_train, y_pred)
    #print('accuracy: ', accuracy, 'loss:',hinge_loss)