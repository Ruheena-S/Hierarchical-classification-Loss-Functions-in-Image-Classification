import os
from typing import Union
import numpy as np
import torch.nn.functional as F  # All functions that don't have any parameters
import pandas as pd
import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torchvision
import torchvision.transforms as transforms  # Transformations we can perform on our dataset
from pandas import io

# from skimage import io
from torch.utils.data import (
    Dataset,
    DataLoader,
)  # Gives easier dataset managment and creates mini batches
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
#from dask import dataframe as df
from tqdm.auto import tqdm
from torch.utils.data.sampler import SubsetRandomSampler
from new_hierarchy import labels,level,get_children,get_parent,get_ancestor,get_descendants,tree_loss,get_level

#if train_model = True then train the model otherwise test the model
train_model = True

#contains the location of the saved model or the model to be saved
model_location = "savedmodels/extracted_short_imagenet/checkpoints/crossentropy(1layer).pth.tar"

def tLoss(output,target):
  batch_size = target.size(0)
  num_classes = output.size(1)
  # Height of the hierarchy tree
  val = None
  pred = None
  loss = 0
  global labels
  probabilities = torch.exp(output)
  #print("target = ",target)
  pred = probabilities.max(dim=1)[1]
  #print("pred = ",pred)
  for i in range(batch_size):
    t = target[i]
    #print("target = ",labels[t])
    #print("pred = ",labels[pred[i]])
    loss = loss + tree_loss(labels[t],labels[pred[i]])
  return loss/batch_size

# Create Fully Connected Network
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, num_classes)
        #self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        #x = F.relu(self.fc1(x))
        x = self.fc1(x)
        return x

class SoloDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        data = np.load(path)
        x_data = torch.tensor(data[0,:])
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))
        return (x_data.float(), y_label)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
num_classes = 218
learning_rate = 0.0001
batch_size = 16
num_epochs = 15
input_size = 2048

# Load Data
train_data = SoloDataset(
    csv_file="data/extracted_short_imagenet/train_labels.csv", root_dir="data/extracted_short_imagenet/train", transform=transforms.ToTensor()
)

test_data = SoloDataset(
    csv_file="data/extracted_short_imagenet/val_labels.csv", root_dir="data/extracted_short_imagenet/val", transform=transforms.ToTensor()
)

print("dataload step1 complete")
#splitting the traindata into train and validation
num_train = len(train_data)
indices = list(range(num_train))
split = int(.2*num_train)			#10% of the data is used for validation
train_idx, valid_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

print("loading the data into loaders")
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler,pin_memory=True)

val_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler,pin_memory=True)

test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, pin_memory=True)

print("len of train_loader: ",len(train_loader))
print("len of val_loader: ",len(val_loader))
print("len of test_loader: ",len(test_loader))

print("data loaded successfully")
# Model
model = NN(input_size=input_size, num_classes=num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = 3, verbose = True)

def save_model():
  checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
              }
  torch.save(checkpoint, model_location)
  print("model saved successfully!!!!!!!")

def load_model():
  checkpoint=torch.load(model_location)
  model.load_state_dict(checkpoint["state_dict"])
  #optimizer.load_state_dict(checkpoint["optimizer"])
  print("Model loaded Successfully")

print(model)

def validation(model, validateloader, criterion):
    
    val_loss = 0
    accuracy = 0
    tl=0
    for images, target in iter(validateloader):

        images, target = images.to(device), target.to(device)

        output = model.forward(images)
        val_loss += criterion(output, target).item()

        probabilities = torch.exp(output)
        
        equality = (target.data == probabilities.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

        tl += tLoss(output.data,target)
    
    return val_loss, accuracy, tl

num_epochs = 15
def train_classifier():

        _epoch = []
        steps = 0
        print_every = 40
        train_error = []
        val_error = []

        for e in range(num_epochs):
            _epoch.append(e+1)
            model.train()
    
            running_loss = 0

            train_accuracy = 0
            for idx,(images, target) in tqdm(enumerate(train_loader), total=len(train_loader), leave=False):
        
                steps += 1
                #print(images.shape)
                #print(target.shape)
                images= images.to(device)
                target = target.to(device)
                optimizer.zero_grad()
                images = torch.autograd.Variable(images, requires_grad=False)
                target = torch.autograd.Variable(target, requires_grad=False)
                output = model.forward(images)
                #print(output.shape) 
                #print(labels.shape)
                loss = criterion(output, target)
                probabilities = torch.exp(output)
                equality = (target.data == probabilities.max(dim=1)[1])
                train_accuracy += equality.type(torch.FloatTensor).mean()
                loss.backward()
                optimizer.step()
        
                running_loss += loss.item()
            scheduler.step(running_loss/len(train_loader))
            validation_loss = 0
            accuracy = 0
            tree_loss = 0
            model.eval()        
            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
              validation_loss, accuracy, tree_loss = validation(model, test_loader, criterion)
                
            print("Epoch: {}/{}.. ".format(e+1, num_epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/len(train_loader)),
                  "Training Accuracy :{:.3f}.. ".format(train_accuracy/len(train_loader)),
                  "Validation Loss: {:.3f}.. ".format(validation_loss/len(test_loader)),
                  "Validation Accuracy: {:.3f}".format(accuracy/len(test_loader)),
                  "Tree Loss: {:.3f}".format(tree_loss/len(test_loader)))
            train_error.append(running_loss/len(train_loader))
            val_error.append(validation_loss/len(test_loader))
            running_loss = 0
            save_model()

if train_model :
  train_classifier()
else :
  model.eval()
  with torch.no_grad():
    validation_loss, accuracy, tree_loss = validation(model, test_loader, criterion)
  print(validation_loss/len(test_loader))
  print(accuracy/len(test_loader))
  print(tree_loss/len(val_loader))
