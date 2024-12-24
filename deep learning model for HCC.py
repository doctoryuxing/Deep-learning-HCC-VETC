# ResNet50, Densenet121, Vision Transformer and Swin Transformer were used to 
# construct the deep radiomics and pathomics models via transfer learning.
# Model architecture and training code for 2D image classification

#%%
from torch.utils import data
from torchvision.datasets import ImageFolder
import torch
print(torch.__version__)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib  
# matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')

import torchvision
from torchvision import transforms
import torchvision.models as models
import os
from tqdm import tqdm
import shutil
import pandas as pd
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
# from scipy import interp
from itertools import cycle
import itertools
import random

##
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

#Set random seed
setup_seed(30)


train_dir = r'...\train'   
test_dir =  r'...\val'


## data augmentation
train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomResizedCrop(224, scale=(0.6,1.0), ratio=(0.8,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(0.2),
    torchvision.transforms.ColorJitter(brightness=0.5, contrast=0, saturation=0, hue=0),
    torchvision.transforms.ColorJitter(brightness=0, contrast=0.5, saturation=0, hue=0),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

##
train_ds =  torchvision.datasets.ImageFolder(
        train_dir,
        transform=train_transform
    )

test_ds =  torchvision.datasets.ImageFolder(
        test_dir,
        transform=test_transform
    )


#In[7]:

BATCH_SIZE = 32

# In[8]:

train_dl = torch.utils.data.DataLoader(
                            train_ds,
                            batch_size=BATCH_SIZE,
                            shuffle=True
)


test_dl = torch.utils.data.DataLoader(
                            test_ds,
                            batch_size=BATCH_SIZE,
)


model = torchvision.models.resnet50(pretrained=True)# resnet50
model = torchvision.models.densenet121(pretrained=True) # densenet121
model = torchvision.models.vit_b_32(pretrained=True) # vit_b_32
model = torchvision.models.swin_v2_b(pretrained=True) #swin_v2_b

model.parameters


#%%
for param in model.parameters():
    param.requires_grad = False


num_classes = 2 

#resnet50
in_f = model.fc.in_features     
model.fc = nn.Linear(in_features=in_f,out_features= num_classes) 
model  

# densenet121
in_f = model.classifier.in_features
model.classifier = nn.Linear(in_features=in_f, out_features=num_classes)
model  

#vit_b_32
in_f = model.heads[0].in_features   
model.heads[0] = nn.Linear(in_features=in_f, out_features= num_classes) 
model 

# swin_v2_b 
in_f = model.head.in_features   
model.head = nn.Linear(in_f, 2)
model

#%% In[15]:
print(torch.cuda.is_available()) 

if torch.cuda.is_available():    
    num_gpus = torch.cuda.device_count()
    print("GPU num：", num_gpus)
    model.to('cuda')

    
loss_fn = nn.CrossEntropyLoss()


# Decay LR by a factor of 0.1 every 7 epochs
from torch.optim import lr_scheduler

optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.0001) #resnet50  
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.0001) #densenet121
optimizer = torch.optim.Adam(model.heads.parameters(), lr=0.0001) #vit
optimizer = torch.optim.Adam(model.head.parameters(), lr=0.0001)#swin-vit

exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

#%%
def fit(epoch, model, trainloader, testloader):
    correct = 0
    total = 0
    running_loss = 0
    model.train()
    for x, y in trainloader:
        if torch.cuda.is_available():
            x, y = x.to('cuda'), y.to('cuda')
        y_pred = model(x)
        loss = loss_fn(y_pred, y) 
        optimizer.zero_grad()  
        loss.backward()  
        optimizer.step() 
        with torch.no_grad():
            y_pred = torch.argmax(y_pred, dim=1)
            correct += (y_pred == y).sum().item()
            total += y.size(0)
            running_loss += loss.item()
    exp_lr_scheduler.step()  
    epoch_loss = running_loss / len(trainloader.dataset)
    epoch_acc = correct / total

    test_correct = 0
    test_total = 0
    test_running_loss = 0

    model.eval()
    with torch.no_grad():
        for x, y in testloader:
            if torch.cuda.is_available():
                x, y = x.to('cuda'), y.to('cuda')
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            y_pred = torch.argmax(y_pred, dim=1)
            test_correct += (y_pred == y).sum().item()
            test_total += y.size(0)
            test_running_loss += loss.item()

    epoch_test_loss = test_running_loss / len(testloader.dataset)
    epoch_test_acc = test_correct / test_total

    print('epoch: ', epoch,
          'train_loss： ', round(epoch_loss, 3),
          'train_accuracy:', round(epoch_acc, 3),
          'test_loss： ', round(epoch_test_loss, 3),
          'test_accuracy:', round(epoch_test_acc, 3)
             )

    return epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc

epochs = 50

train_loss = []
train_acc = []
test_loss = []
test_acc = []

for epoch in range(epochs):
    epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc = fit(epoch,model,train_dl,test_dl)
    train_loss.append(epoch_loss)
    train_acc.append(epoch_acc)
    test_loss.append(epoch_test_loss)
    test_acc.append(epoch_test_acc)


#%%
for param in model.parameters(): 
    param.requires_grad = True

len(list(model.parameters())) 
list(model.parameters())[-3:] 


for param in list(model.parameters())[-3:]:  
    print(param)
    param.requires_grad = True 

extend_epochs = 50

from torch.optim import lr_scheduler        
optimizer = torch.optim.Adam(list(model.parameters())[-3:], lr=0.0001)  
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

for epoch in range(extend_epochs):
    epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc = fit(epoch,model,train_dl,test_dl)
    train_loss.append(epoch_loss)
    train_acc.append(epoch_acc)
    test_loss.append(epoch_test_loss)
    test_acc.append(epoch_test_acc)


##save model
PATH = './my_model.pth'
torch.save(model.state_dict(), PATH)
