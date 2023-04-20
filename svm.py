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
#from sklearn.model_selection import train_test_split
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


if __name__ == '__main__':
    labels = [35,43,47,53,61,65,71,85]
    args = parse_args()
    trainset = OCTDataset(args, 'train', transform=transform)
    testset = OCTDataset(args, 'test', transform=transform)
     
    # Feature extraction is required before using the SVM model 
    # Extract HOG features for training images 
    os.environ['TORCH_HOME'] = "C:/fml_final_project"

    train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True) 

    # Split the dataset into training and testing sets using encoded data
    #for i, (X_train,y_train) in enumerate(train_dataloader):    
    #X_train, X_val, y_train, y_val = train_test_split(trainset, testset, test_size=0.3, random_state=0)
 

    # split the x_train, x_test, y_train, y_test
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    for i, (x,y) in enumerate(train_dataloader):
        X_train.append(x)
        y_train.append(y)
        print(i,' train')
    for i,(x,y) in enumerate(test_dataloader):
        X_test.append(x)
        y_test.append(y)
        print(i,' test')

    # initiate svm classifier 
    svmclf = SVC(kernel = 'linear')
    # train the svm model
    svmclf.fit(X_train,y_train)
    y_pred = svmclf.predict(X_test)

    # accuracy score 
    accuracy = accuracy_score(y_test, y_pred)
    # loss
    hinge_loss = metrics.hinge_loss(y_train, y_pred)
    print('accuracy: ', accuracy, 'loss:',hinge_loss)
