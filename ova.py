
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_optimizer import Ranger
import torch.optim as optim
from pandas import io
import os
from typing import Union
from torchvision import datasets
from torchvision import transforms,models
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataloader import Dataset
from torch.utils.data import random_split
from torchmetrics import Accuracy
from torch.optim.lr_scheduler import CyclicLR
from tqdm.auto import tqdm

torch.manual_seed(42)

from google.colab import drive
drive.mount('/content/drive')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ",device)


traindataset = '/lfs/usrhome/mtech/cs21m053/scratch/MBM/imagenet/train'
testdataset = '/lfs/usrhome/mtech/cs21m053/scratch/MBM/imagenet/val'

num_classes = 1753     
learning_rate = 1e-3
batch_size = 16
num_epochs = 20

def target_trans(target):
  y = -torch.ones(num_classes)
  y[target] = 1

  global labels
  label = labels[target]
  for j in range(inter_labels):
    if label in get_descendants(labels[leaf_labels+j]):
      y[leaf_labels+j] = 1

  return y,target
  
  
class transform_Dataset(Dataset):
    def __init__(self, csv_file, path1, path2, transform=None,target_transform=None):
        self.annotations = pd.read_csv(csv_file,header=None)
        self.path1 = path1
        self.file = np.load(path2)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        path = os.path.join(self.path1, self.annotations.iloc[index, 0])
        data = np.load(path)
        x_data = torch.tensor(data[0,:])
        y_label = (torch.tensor(self.file[int(self.annotations.iloc[index,1])],dtype=torch.int64),int(self.annotations.iloc[index,1]))
        return (x_data.float(), y_label)
  
# Load Data
train_data = transform_Dataset(
    csv_file = "/lfs/usrhome/mtech/cs21m053/scratch/MBM/imagenet/train_labels.csv",
    path1 = "/lfs/usrhome/mtech/cs21m053/scratch/MBM/imagenet/train",
    path2 = "/scratch/scratch6/CS19M010/extracted_full_imagenet/ova_target_labels.npy",
    transform = transforms.ToTensor(),
    target_transform = None)

test_data = transform_Dataset(
    csv_file = "/lfs/usrhome/mtech/cs21m053/scratch/MBM/imagenet/val_labels.csv",
    path1 = "/lfs/usrhome/mtech/cs21m053/scratch/MBM/imagenet/val",
    path2 = "/scratch/scratch6/CS19M010/extracted_full_imagenet/ova_target_labels.npy",
    transform=transforms.ToTensor(),
    target_transform = None)

# Data split - 0.8 for training and 0.2 for validation
num_train = len(train_data)
indices = list(range(num_train))
split = int(.2*num_train)
			
train_idx, valid_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

#loading the data into loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler,pin_memory=True)

val_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler,pin_memory=True)

test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, pin_memory=True)
  
  


# Input    - output : Predicted output and target : True output of the input
# Function - For each batch of inputs, finds the tree loss using OvA algorithm.
# output   - Returns the floating integer by calculating the batch-wise tree loss

def tLoss(output,target,h=14):

  batch_size = target.size(0)
  num_classes = output.size(1)
  
  tau_values=[0.5]*h

  val = None
  pred = None
  loss = 0
  global labels

  # Iterate each example in the batch
  for i in range(batch_size):
    t = target[i]
    
    # searching in the bottom-up approach in the hierarchy tree
    h = 14
    while h >= 0:
    
      # Clone the output array each time for checking at every level 
      values = output.clone().detach()
      for j in range(num_classes):
        level = get_level(labels[j])
        
        # Values of nodes below the current level are considered in the ancestor hence we assign the min value 
        if level > h:
          values[i,j] = -10000
          
        elif level < h:
          # Similary for the node which is above current level but is not a leaf node, we assign the min value
          if get_children(labels[j]) is not None:
            values[i,j] = -10000      
      val,pred = torch.max(values[i,:],0)
      
      # If the output is greater than the tau value at each level we consider the pred value assigned in the previous step
      if val >= tau_values[h]:
        break
        
      h = h - 1

    # Total loss for the batch is calculated
    loss = loss + tree_loss(labels[t],labels[pred])

  loss = loss/batch_size

  # Average loss for the batch is returned
  return loss



class Hinge_Loss(torch.nn.Module):
    
    def __init__(self):
        super(Hinge_Loss,self).__init__()
        
    def forward(self,x,y):

        temp = 1 - x * y
        #print(temp.shape)
        zero = torch.zeros(x.size()[0],x.size()[1])
        zero = zero.to(device)
        clamp = torch.max(temp,zero)
        #print(clamp.shape)
        total_loss = torch.sum(clamp)/x.size()[0]
        return total_loss


# Define the ResNet50 model and freeze layers

model = models.resnet50(pretrained=True)
model.conv1 = nn.Conv2d(3, 64, kernel_size = (3,3), padding = (1, 1), bias = False)
model.maxpool = nn.Identity()

for param in model.parameters():
    param.requires_grad = True
# for param in model.layer4.parameters():
#     param.requires_grad = True

model.fc = nn.Linear(2048, num_classes)

model.to(device)



# Define loss function and optimizer

criterion = Hinge_Loss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = 3, verbose = True)



def get_accuracy(output, target):

  batch_size = target.size(0)

  pred = output[:,0:1000] # considering the first 1000 of the 1753 nodes because they are the true classes(No internal node)
  pred = pred.max(dim=1)[1]
  correct = pred==target

  acc = correct.float().sum(0)
  return acc/batch_size
  
#Training

total_train_step = len(train_loader)
#print(total_train_step)
total_val_step=len(valid_loader)
BEST_VAL_METRIC = 0
BEST_MODEL = None


for epoch in range(1, num_epochs+1):

    train_loss=0
    train_acc=0.0
    model.train()

    for i, (images, target) in enumerate(train_loader, 1):

        y_trans = target[0]
        y_true = target[1]

        # Move tensors to the configured device
        images = images.to(device)
        y_true = y_true.to(device)
        y_trans = y_trans.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, y_trans)

        train_loss += loss
        train_acc += get_accuracy(outputs, y_true)
        #train_acc += accuracy(outputs[:,0:100], y_true)
        
        
        # Backward and optimize
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # scheduler.step(train_loss/total_train_step)
        

    print(f'Epoch [{epoch}/{num_epochs}] - Loss: {(train_loss/total_train_step):.4f}, Accuracy: {(train_acc/total_train_step):.4f}')

    model.eval() 
    # Validation
    with torch.no_grad():
        val_acc = 0
        val_loss=0
        Tree_Loss_Value = 0
        for i, (images, target) in enumerate(valid_loader, 1):

            y_trans = target[0]
            y_true = target[1]

            # Move tensors to the configured device
            images = images.to(device)
            y_true = y_true.to(device)
            y_trans = y_trans.to(device)

            outputs = model(images)
            val_loss += criterion(outputs, y_trans)
            Tree_Loss_Value += tLoss(outputs,y_true)
            val_acc += get_accuracy(outputs, y_true)
            #val_acc += accuracy(outputs[:,0:100], y_true)

    if val_acc/total_val_step > BEST_VAL_METRIC:
        BEST_VAL_METRIC = val_acc/total_val_step
        BEST_MODEL = model.state_dict() 
        torch.save(model, "/lfs/usrhome/mtech/cs21m053/scratch/MBM/out/bestcheckpoint.pth")

    print(f'Accuracy of the network on validation images: {(val_acc/total_val_step):.4f}, loss: {(val_loss/total_val_step):.4f}, Tree loss: {(Tree_Loss_Value/total_val_step):.4f}') 


#Testing
model.load_state_dict(BEST_MODEL)

total_test_step=len(test_loader)

with torch.no_grad():
    test_acc=0
    test_loss=0
    Tree_Loss_Value=0

    for i, (images, target) in enumerate(test_loader, 1):
        
        y_trans = target[0]
        y_true = target[1]
        
        images = images.to(device)
        y_true = y_true.to(device)
        y_trans = y_trans.to(device)

        # Forward pass
        outputs = model(images)
        
        # Loss
        test_loss += criterion(outputs,y_trans)
        Tree_Loss_Value += tLoss(outputs,y_true)
        test_acc += get_accuracy(outputs, y_true)
        #test_acc += accuracy(outputs[:,0:100], y_true)

    print(f'Accuracy of the network on test images: {(test_acc/total_test_step):.4f}, loss: {(test_loss/total_test_step):.4f}, Tree loss: {(Tree_Loss_Value/total_test_step):.4f}')
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     

