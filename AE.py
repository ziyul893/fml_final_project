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
#from skimage.feature import hog



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
            img = self.transform(img) # reshape

        return img, target

    def __len__(self):
        return len(self._labels)         

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annot_train_prime', type = str, default = 'df_prime_train.csv')
    parser.add_argument('--annot_test_prime', type = str, default = 'df_prime_test.csv')
    parser.add_argument('--data_root', type = str, default = 'C:/fml_final_project')
    return parser.parse_args()

# Define the autoencoder architecture
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        ## encoder layers ##
        # conv layer (depth from 3 --> 16), 3x3 kernels
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)  
        # conv layer (depth from 16 --> 4), 3x3 kernels
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.pool = nn.MaxPool2d(2, 2)
        
        ## decoder layers ##
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 1, 2, stride=2)

    def forward(self, x):
        ## encode ##
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        # add second hidden layer
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # compressed representation
        
        ## decode ##
        # add transpose conv layers, with relu activation function
        x = F.relu(self.t_conv1(x))
        # output layer (with sigmoid for scaling from 0 to 1)
        x = F.sigmoid(self.t_conv2(x))
        return x


if __name__ == '__main__':
    labels = [35,43,47,53,61,65,71,85]
    args = parse_args()
    trainset = OCTDataset(args, 'train', transform=transform)
    testset = OCTDataset(args, 'test', transform=transform)
    # print(trainset[1][0].shape)
    # print(len(trainset), len(testset))
     
    # Feature extraction is required before using the SVM model 
    # Extract HOG features for training images 
    os.environ['TORCH_HOME'] = "C:/fml_final_project"
    #print(torch.cuda.is_available())

    train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True) 

    '''
    X_train = []
    y_train = []
    for i in range(len(trainset)):
        img, target = trainset[i]
        features = hog(img.numpy().squeeze(), orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), transform_sqrt=True, feature_vector=True)
        X_train.append(features)
        y_train.append(target)

    # Extract HOG features for testing images 
    X_test = []
    y_test = []
    for i in range(len(testset)):
        img, target = testset[i]
        features = hog(img.numpy().squeeze(), orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), transform_sqrt=True, feature_vector=True)
        X_test.append(features)
        y_test.append(target)

'''
    # use autoencoder to convert 1x224x224 img to 2 dimensions
    ae = Autoencoder().to('cuda')
    #encoded_data,decoded_data = ae.forward(X_train)
    
    # train the autoencoder 
    optimizer = torch.optim.Adam(ae.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    num_epochs = 5

    
    for epoch in range(num_epochs):
        for i, (x,y) in enumerate(train_dataloader):    
            ae.train()
            #print('x size is:',x.shape)
            optimizer.zero_grad()      
            # extract input, obtain model output, loss, and backpropagate
            #x = torch.tensor(x)
            #y = torch.tensor(y)

            input=x.cuda()
            #input = torch.tensor(input)

            #target= input
            #print (type(input))
            #print(input.shape)
            #print (type(target))
            #print(target.shape)
            out = ae(input)
            #out = torch.tensor(out)
            #print(type(out))
            #print(out.shape)
            loss_val=loss_fn(input, out).cuda()
            loss_val.backward()
            optimizer.step()
            #print("done",i)

        print('Epoch: {} | Loss:{:0.6f}'.format(epoch, loss_val.item()))
    torch.save(ae.state_dict(), 'autoencoder_model.pth')
    # Split the dataset into training and testing sets using encoded data
    X_train, X_val, y_train, y_val = train_test_split(encoded_data, y_train, test_size=0.3, random_state=0)

    # initiate svm classifier 
    svmclf = svm.SVC(kernel = 'linear')
    # train the svm model
    svmclf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)

    # accuracy score 
    accuracy = accuracy_score(y_test, y_pred)
    # loss
    hinge_loss = metrics.hinge_loss(y_train, y_pred)
