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
        self.eye_data = self.annot['Eye_Data'].values
        self._labels = self.annot['Severity_Label'].values
        assert len(self.path_list) == len(self._labels)
        # idx_each_class = [[] for i in range(self.nb_classes)]

    def __getitem__(self, index):
        img, target, eye = Image.open(self.root+self.path_list[index]).convert("L"), self._labels[index], self.eye_data[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, target, eye

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
    epochs  = 30
    lr = 0.001

    args = parse_args()
    trainset = OCTDataset(args, 'train', transform=transform)
    testset = OCTDataset(args, 'test', transform=transform)
    print(trainset[1][0].shape)
    print(len(trainset), len(testset))
    os.environ['TORCH_HOME'] = "S:/8803Proj"
    print(torch.cuda.is_available())
    #models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
    #models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    train_dataloader = DataLoader(trainset, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(testset, batch_size=1, shuffle=True)

    model = models.alexnet().cuda()
    model.features[0] = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False).cuda()
    model.classifier[6] = nn.Linear(4096,3).cuda()

    # define loss function
    loss = nn.CrossEntropyLoss()

    # set up optimizer with the specified learning rate
    optimizer = torch.optim.SGD(model.parameters(),lr=lr, momentum=0.9)

    for epoch in range(epochs): #flashbulb training with epoch 5
        if epoch == 5:
            optimizer.param_groups[0]['lr'] = 0.001
            for i, (x, y, z) in enumerate(train_dataloader):
                if(y.cuda().item() == 1):
                    model.train()
                    optimizer.zero_grad()

                     # extract input, obtain model output, loss, and backpropagate
                    input=x.cuda()
                    input = (input - input.min())/(input.max()-input.min())
                    target=y.cuda()
                            
                    out = model(input)
                    loss_val=loss(out,target).cuda()
                    loss_val.backward()
                    optimizer.step()
            optimizer.param_groups[0]['lr'] = 0.001
        else: 
            for i, (x, y, z) in enumerate(train_dataloader):
                if(y.cuda().item() == 0)or (y.cuda().item() == 2):
              
                    model.train()
                    optimizer.zero_grad()

                    # extract input, obtain model output, loss, and backpropagate
                    input=x.cuda()
                    input = (input - input.min())/(input.max()-input.min())
                    target=y.cuda()
                                    
                    out = model(input)
                    loss_val=loss(out,target).cuda()
                    loss_val.backward()
                    optimizer.step()


        print('Epoch: {} | Loss:{:0.6f}'.format(epoch, loss_val.item()), loss_val.item())

    
    model_scripted = torch.jit.script(model) # Export to TorchScript
    model_scripted.save('AlexNet_notnormal_no1_5_lr007.pt') # save to file, name is dynamic
    
    
