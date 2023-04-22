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
from torch.autograd import Variable
import torch.nn.functional as F
from glob import glob

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

def imread(img_dir):
  # read the images into a list of `PIL.Image` objects
  images = []
  for f in glob(os.path.join(img_dir, "*")):
    images.append(Image.open(f).convert('RGB'))

  return images

def vis_img_label(image_list, label_list=None):
  # visualize the images w/ labels
  Tot = len(image_list)
  Cols = 4
  Rows = Tot // Cols 
  Rows += (Tot % Cols)>0
  if label_list is None:
    label_list = [""]*Tot
  # Create a Position index
  Position = range(1,Tot + 1)
  fig = plt.figure(figsize=(Cols*5, Rows*5))
  for i in range(Tot):
    image = image_list[i]
    # add every single subplot to the figure 
    ax = fig.add_subplot(Rows,Cols,Position[i])
    ax.imshow(np.asarray(image))    
    ax.set_title(label_list[i])

class Hook():

  def __init__(self, ):
    self.z_grad = None
    self.y_grad = None
  
  # hook for z
  def z_backward_hook(self, grad):
    #grad.requires_grad = True
    self.z_grad = grad#.retain_grad()
    
    return None

  # hook for y
  def y_backward_hook(self, grad):
    #grad.requires_grad = True
    self.y_grad = grad#.retain_grad()

    return None

  def register(self, y, z):

    # register hooks
    z.register_hook(self.z_backward_hook)
    y.register_hook(self.y_backward_hook)

class GradCAM():

  def __init__(self, model_object, layer_idx):
    """
    Args:
        model_object: model object that we apply Grad-CAM with
        layer_idx: index of the layer which we get activation maps

    """
      
    self.model = model_object

    self.gradients = dict()
    self.activations = dict()

    # backward hook for gradient
    def backward_hook(module, grad_input, grad_output):
      self.gradients['value'] = grad_output[0] 
      return None

    # forward hook for activation map
    def forward_hook(module, input, output):
      self.activations['value'] = output
      return None

    target_layer = self.model.fc

    # register hooks
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)

  def calculate(self, input):
    """
    Args:
        input: input image with shape of (1, 3, H, W)

    Return:
        saliency_map: saliency map of the same spatial dimension with input
        logit: model output
    """
    b, c, h, w = input.size()
    self.model.eval()
    self.model.cuda()

    # model output
    logit = self.model(input)

    # predicting class
    max = logit.max(1)[-1]
    y_c = logit[:,max].squeeze()

    self.model.zero_grad()
    y_c.backward()

    # get activation maps and gradients
    gradients = self.gradients['value']
    activations = self.activations['value']
    b, k= gradients.size()

    # calculate alpha (step1)
    alpha = np.mean(gradients.reshape((b, k, -1)),2)

    # calculate Grad-CAM result using rectified weighted linear combination of feature activation maps (step2)
    ##TODO
    weight = alpha.reshape((b, k, 1, 1))
    saliency_map = (weight*activations).sum(1, keepdim=True)
    saliency_map = F.relu(saliency_map)

    saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
    saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
    saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data

    return saliency_map, logit

  def __call__(self, input):
    return self.calculate(input)

def visualize_cam(mask, img):
    
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = heatmap.astype('float64') / 255
    heatmap = heatmap[:, :, [2, 1, 0]]
    
    result = heatmap + np.asarray(img).astype('float64') / 255
    result = result.astype('float64') / np.amax(result)
    
    return heatmap, result

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annot_train_prime', type = str, default = 'df_prime_train.csv')
    parser.add_argument('--annot_test_prime', type = str, default = 'df_prime_test.csv')
    parser.add_argument('--data_root', type = str, default = 'S:/8803Proj')
    return parser.parse_args()

if __name__ == '__main__':
    labels = [35,43,47,53,61,65,71,85]
    epochs  = 1
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

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).cuda()
    for param in model.parameters():
        param.requires_grad = False
    num_features = model.fc.in_features
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False).cuda()
    model.fc = nn.Linear(num_features,3).cuda()
    #summary(model,(1,224,224))

    # define loss function
    loss = nn.CrossEntropyLoss()

    # set up optimizer with the specified learning rate
    optimizer = torch.optim.SGD(model.fc.parameters(),lr=lr, momentum=0.9)

    for epoch in range(epochs): #flashbulb training on epoch 15
        if epoch == 15:
            optimizer = torch.optim.SGD(model.fc.parameters(),lr=1, momentum=0.9)
            for i, (x, y, z) in enumerate(train_dataloader):
                if(y.cuda().item() == 1):
                    model.train()
                    optimizer.zero_grad()

                            # extract input, obtain model output, loss, and backpropagate
                    input=x.cuda()
                    target=y.cuda()
                            
                    out = model(input)
                    loss_val=loss(out,target).cuda()
                    loss_val.backward()
                    optimizer.step()
                
                    
                    #print(z)
                    #print(y.cuda().item())
                    #if i>2:
                    #    break
            optimizer = torch.optim.SGD(model.fc.parameters(),lr=lr, momentum=0.9)
        else: 
            for i, (x, y, z) in enumerate(train_dataloader):
                if(y.cuda().item() == 0)or (y.cuda().item() == 2):
                    model.train()
                    optimizer.zero_grad()

                            # extract input, obtain model output, loss, and backpropagate
                    input=x.cuda()
                    target=y.cuda()
                            
                    out = model(input)
                    loss_val=loss(out,target).cuda()
                    loss_val.backward()
                    optimizer.step()
                
                    
                    #print(z)
                    #print(y.cuda().item())
                    #if i>2:
                    #    break

        print('Epoch: {} | Loss:{:0.6f}'.format(epoch, loss_val.item()))

    
    model_scripted = torch.jit.script(model) # Export to TorchScript
    model_scripted.save('hi.pt') # Save
    

