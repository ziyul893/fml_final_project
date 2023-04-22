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
import Train_ResNet
import torch.nn.functional as F
from sklearn.svm import SVC
import sklearn
from sklearn.naive_bayes import GaussianNB
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
    args = Train_ResNet.parse_args()
    
    trainset = Train_ResNet.OCTDataset(args, 'train', transform=transform)
    testset = Train_ResNet.OCTDataset(args, 'test', transform=transform)

    train_dataloader = DataLoader(trainset, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(testset, batch_size=1, shuffle=True)

    model = torch.jit.load('AE_for_visual_finals.pt')
    '''
    #model.eval()
    appeared = [0, 0, 0]
    correct = [0, 0, 0]
    pred = [0,0,0]
    
    class0_x = []
    class0_y = []
    
    class1_x = []
    class1_y = []
    
    class2_x = []
    class2_y = []
    with torch.no_grad():
        for i in range(7987):
            img, target, eye = next(iter(test_dataloader)) # extract image from train loader
            img = img.cuda()#.permute(1,0,2,3)
            target = target.cuda().item()
            out, rep = model(img)# extract output vector using model
            num = rep.cpu().numpy()[0][0][1]
            if target == 0:
                class0_x.append(num[0])
                class0_y.append(num[1])
                
            elif target == 1:
                class1_x.append(num[0])
                class1_y.append(num[1])
                
            elif target == 2:
                class2_x.append(num[0])
                class2_y.append(num[1])
                #print()
            #print(rep.cpu().numpy())

    plt.plot(class0_x, class0_y, c='r', label = "class 0")
    plt.plot(class1_x, class1_y, c='g', label = "class 1")
    plt.plot(class2_x, class2_y, c='b', label = "class 2")
    plt.legend()
    plt.show()
    '''
    # split the x_train, x_test, y_train, y_test
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    
    for i, (x,y,z) in enumerate(train_dataloader):
        x = x.cuda()
        out, rep = model(x)
        rep = rep.detach().cpu().numpy()
        X_train.append(rep.reshape(1,-1))
        y_train.append(y.item())
        #print(i,' train')
    for i,(x,y,z) in enumerate(test_dataloader):
        x = x.cuda()
        out, rep = model(x)
        X_test.append(rep.detach().cpu().numpy().reshape(1,-1))
        y_test.append(y.item())
        #print(i,' test')
    

    X_test = np.array(X_test).squeeze()
    X_train = np.array(X_train).squeeze()
    y_train = np.array(y_train).squeeze()
    y_test = np.array(y_test).squeeze()

    gnb = GaussianNB() #Naive Bayes Classifier
    y_pred = gnb.fit(X_train, y_train).predict(X_test)

    ''' Knn classifier
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train).predict(X_test)
    '''


    # accuracy score 
    accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
    # loss
    #hinge_loss = sklearn.metrics.hinge_loss(y_test, y_pred)
    print('accuracy: ', accuracy)#, 'loss:',hinge_loss)
