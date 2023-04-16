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
    
class OCTDataset(Dataset):
    def __init__(self, args, subset='train', transform=None,):
        if subset == 'train':
            self.annot = pd.read_csv(args.annot_train_prime)
        elif subset == 'test':
            self.annot = pd.read_csv(args.annot_test_prime)
            
        self.annot['Severity_Label'] = [LABELS_Severity[drss] for drss in copy.deepcopy(self.annot['DRSS'].values)] 
        # print(self.annot)
        self.root = os.path.expanduser(args.data_root)
        self.transform = transform
        # self.subset = subset
        self.nb_classes=len(np.unique(list(LABELS_Severity.values())))
        self.path_list = self.annot['File_Path'].values
        self._labels = self.annot['Severity_Label'].values
        assert len(self.path_list) == len(self._labels)
        # idx_each_class = [[] for i in range(self.nb_classes)]

    def __getitem__(self, index):
        img, target = Image.open(self.root+self.path_list[index]).convert("L"), self._labels[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self._labels)         


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annot_train_prime', type = str, default = 'df_prime_train.csv')
    parser.add_argument('--annot_test_prime', type = str, default = 'df_prime_test.csv')
    parser.add_argument('--data_root', type = str, default = 'S:/8803Proj')
    return parser.parse_args()

if __name__ == '__main__':
    labels = [35,43,47,53,61,65,71,85]
    epochs  = 2
    lr = 0.0001

    args = parse_args()
    trainset = OCTDataset(args, 'train', transform=transform)
    testset = OCTDataset(args, 'test', transform=transform)
    print(trainset[1][0].shape)
    print(len(trainset), len(testset))
    os.environ['TORCH_HOME'] = "S:/8803Proj"
    print(torch.cuda.is_available())
    #models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
    #models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    train_dataloader = DataLoader(trainset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(testset, batch_size=1, shuffle=True)

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).cuda()
    for param in model.parameters():
        param.requires_grad = False
    num_features = model.fc.in_features
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False).cuda()
    model.fc = nn.Linear(num_features,8).cuda()
    #summary(model,(1,224,224))

    # define loss function
    loss = nn.CrossEntropyLoss()

    # set up optimizer with the specified learning rate
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)

    for epoch in range(epochs):
        for i, (x, y) in enumerate(train_dataloader):
            
            model.train()
            optimizer.zero_grad()

            # extract input, obtain model output, loss, and backpropagate
            input=x.cuda()
            target=y.cuda()

            out = model(input)
            loss_val=loss(out,target).cuda()
            loss_val.backward()
            optimizer.step()
            print("done",i)
            #if i>2:
            #    break

        print('Epoch: {} | Loss:{:0.6f}'.format(epoch, loss_val.item()))

    correct = 0
    with torch.no_grad():
        for i in range(500):
            img, target = next(iter(test_dataloader)) # extract image from train loader
            img = img.cuda()
            target = target.cuda().item()
            out = model(img)# extract output vector using model
            print()
            pred_label = LABELS_Severity[labels[out.argmax().item()]] # obtain predicted label
            print("preded: ", pred_label)
            print("actual: ", target)
            if pred_label == target:
                correct+=1
    
    print(correct)

