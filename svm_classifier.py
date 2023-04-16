import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import pandas as pd
from PIL import Image
import argparse
import os
import copy


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

# resize to (3,224,224)
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
    parser.add_argument('--data_root', type = str, default = '')
    return parser.parse_args()

# Define the autoencoder architecture
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2, 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(4, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


if __name__ == '__main__':
    args = parse_args()
    trainset = OCTDataset(args, 'train', transform=transform)
    testset = OCTDataset(args, 'test', transform=transform)
    # print(trainset[1][0].shape)
    # print(len(trainset), len(testset))
     
    # Feature extraction is required before using the SVM model 
    # Extract HOG features for training images 
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

    # use autoencoder to convert 1x224x224 img to 2 dimensions
    ae = Autoencoder()
    encoded_data,decoded_data = ae.forward(X_train)
   
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
