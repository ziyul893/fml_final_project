import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import pandas as pd
from PIL import Image
import argparse
import os
import copy
import torch.nn as nn
import torch.nn.functional as F
import matplotlib as plt

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

# resize to (1,224,224)
transform = transforms.Compose([
    transforms.Resize(size=(256,512)),
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
            img = self.transform(img) # reshape

        return img, target

    def __len__(self):
        return len(self._labels)         

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annot_train_prime', type = str, default = 'df_prime_train.csv')
    parser.add_argument('--annot_test_prime', type = str, default = 'df_prime_test.csv')
    parser.add_argument('--data_root', type = str, default = 'S:/8803Proj')
    return parser.parse_args()

# Define the autoencoder architecture
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        ## encoder layers ##
        # conv layer (depth from 1 --> 8), 3x3 kernels
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)  
        # conv layer (depth from 8 --> 4), 3x3 kernels
        self.conv2 = nn.Conv2d(8, 4, 3, padding=1)
        # conv layer (depth from 4 --> 2), 3x3 kernels
        self.conv3 = nn.Conv2d(4, 2, 3, padding=1)
        # conv layer (depth from 2 --> 1), 3x3 kernels
        self.conv4 = nn.Conv2d(2, 1, 3, padding=1)
        # pooling layer to reduce x-y dims by two
        self.pool = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(4, 4)
        self.pool3 = nn.MaxPool2d(8, 8)
        self.pool4 = nn.MaxPool2d(8, 4)
        
        ## decoder layers ##
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.t_conv2 = nn.ConvTranspose2d(1, 2, 8, stride=8)
        self.t_conv3 = nn.ConvTranspose2d(2, 4, 4, stride=4)
        self.t_conv4 = nn.ConvTranspose2d(4, 8, 4, stride=4)
        self.t_conv5 = nn.ConvTranspose2d(8, 1, 2, stride=2)

    def forward(self, x):
        ## encode ##
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        # add second hidden layer
        # 128
        x = F.relu(self.conv2(x))
        x = self.pool2(x)  # compressed representation
        # 32
        x = F.relu(self.conv3(x))
        x = self.pool2(x)  # compressed representation
        # 8
        x = F.relu(self.conv4(x))
        x = self.pool3(x)  # compressed representation
        
        lower_rep = x
        
        ## decode ##
        # add transpose conv layers, with relu activation function after each conv layer 
        x = F.relu(self.t_conv2(x))
        x = F.relu(self.t_conv3(x))
        x = F.relu(self.t_conv4(x))
        # output layer (with sigmoid for scaling from 0 to 1)
        x = F.sigmoid(self.t_conv5(x))
        return x, lower_rep


if __name__ == '__main__':
    labels = [35,43,47,53,61,65,71,85]
    args = parse_args()
    trainset = OCTDataset(args, 'train', transform=transform)
    testset = OCTDataset(args, 'test', transform=transform)
     
    # Feature extraction is required before using the SVM model 
    # Extract HOG features for training images 
    os.environ['TORCH_HOME'] = "S:/8803Proj"
    #print(torch.cuda.is_available()) # check the environment 

    train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True) 

    # use autoencoder to convert 1x224x224 img to 2 dimensions
    ae = Autoencoder().to('cuda')
    #encoded_data,decoded_data = ae.forward(X_train)
    
    # train the autoencoder 
    optimizer = torch.optim.Adam(ae.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    num_epochs = 50

    
    for epoch in range(num_epochs):
        for i, (x,y) in enumerate(train_dataloader):    
            ae.train()
            optimizer.zero_grad()      
            input=x.cuda()#.permute(1,0,2,3)
            input = (input - input.min())/(input.max()-input.min())
            out, rep = ae(input)
            print(rep.detach().cpu().numpy()[0][0])
            loss_val=loss_fn(input, out).cuda()
            loss_val.backward()
            optimizer.step()
            #print("done",i)

        print('Epoch: {} | Loss:{:0.6f}'.format(epoch, loss_val.item()))
        
    model_scripted = torch.jit.script(ae) # Export to TorchScript
    model_scripted.save('AE_for_visual_finals.pt') # Save
