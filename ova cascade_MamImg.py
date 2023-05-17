--import os
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ",device)

#speed up things a little bit
#cudnn.benchmark = True
# flag to train the model   if train_model = true then train the model else test the model
train_model = True

#contains path to the save_model or model to be saved
model_location = "savedmodels/extracted_short_imagenet/checkpoints/ova(2layer).pth.tar"

#path to directories containing train and test data
data_dir = r'/data/extracted_short_imagenet'
train_dir = data_dir + '/train'
test_dir = data_dir + '/val'
print("train_dir : ",train_dir)
print("test_dir : ",test_dir)


# function to calculate tree loss using OvA cascade prediction 
def tLoss(output,target,tau_values=[0,0,0,0,0,0,0,0,0,0]):

  batch_size = target.size(0)
  num_classes = output.size(1)
  # Height of the hierarchy tree
  h = 9
  val = None
  pred = None
  loss = 0
  global labels

  # Loop for each example in the batch
  for i in range(batch_size):
    t = target[i]
    # Go searching bottom-up in the hierarchy tree
    h = 9
    while h >= 0:
      # Clone the output array each time for checking at every level in the bottom-up search 
      values = output.clone().detach()
      for j in range(num_classes):
        level = get_level(labels[j])
        # All the nodes below the current level of search are already absorbed into the ancestor of that node at the current level
        # So assigning the value for that node a huge negative value so that it does not affect searching for the max function at the level 
        if level > h:
          values[i,j] = -10000
        elif level < h:
          # Also, if the node is above current level in the hierarchy tree but is not a leaf node, we don't have any examples for that class 
          # So we assign huge negative value so it does not affect the finding for the max
          if get_children(labels[j]) is not None:
            values[i,j] = -10000      
      val,pred = torch.max(values[i,:],0)
      # Checking if the output is greater than the tau value at each level in the hierarchy bottom-up
      # If greater, we break and the predicted label is pred
      if h == 9:
        if val >= tau_values[9]:
          break
      if h == 8:
        if val >= tau_values[8]:
          break
      if h == 7:
        if val >= tau_values[7]:
          break
      if h == 6:
        if val >= tau_values[6]:
          break
      if h == 5:
        if val >= tau_values[5]:
          break
      if h == 4:
        if val >= tau_values[4]:
          break
      if h == 3:
        if val >= tau_values[3]:
          break
      if h == 2:
        if val >= tau_values[2]:
          break
      if h == 1:
        if val >= tau_values[1]:
          break
      if h == 0:
        if val >= tau_values[0]:
          break
      h = h - 1
    # tree_loss() function gives the tree distance between the two nodes
    # Total loss for the batch is calculated
    loss = loss + tree_loss(labels[t],labels[pred])

  loss = loss/batch_size

  # Average loss for the batch is returned
  return loss

#function to translate the target labels according to the model
def target_trans(target):
  y = -torch.ones(323)
  y[target] = 1

  global labels
  label = labels[target]
  for j in range(105):
    if label in get_descendants(labels[218+j]):
      y[218+j] = 1

  return y,target

class SoloDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None,target_transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        data = np.load(path)
        x_data = torch.tensor(data[0,:])
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))
        y_label = self.target_transform(y_label)
        return (x_data.float(), y_label)

num_classes = 323    #total number of nodes in the hierarchy
learning_rate = 1e-3
batch_size = 16
num_epochs = 30
input_size = 2048

# Load Data
train_data = SoloDataset(
    csv_file="data/extracted_short_imagenet/train_labels.csv", 
    root_dir=train_dir, 
    transform=transforms.ToTensor(), 
    target_transform = target_trans)

test_data = SoloDataset(
    csv_file="data/extracted_short_imagenet/val_labels.csv", 
    root_dir=test_dir, 
    transform=transforms.ToTensor(),
    target_transform=target_trans)

#splitting the traindata into train and validation
num_train = len(train_data)
indices = list(range(num_train))
split = int(.2*num_train)			#20% of the data is used for validation
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

# Create Fully Connected Network
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#Loss function
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


# Model
model = NN(input_size=input_size, num_classes=num_classes).to(device)

# Loss and optimizer
criterion = Hinge_Loss()
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

load_model()

def get_accuracy(output, target):
  """Computes the precision@k for the specified values of k"""

  batch_size = target.size(0)

  # We only take first 100 of the 127 nodes because only the first 100 nodes are the classifiers on the true classes(No internal node)
  pred = output[:,0:218]
  pred = pred.max(dim=1)[1]
  correct = pred==target

  acc = correct.float().sum(0)
  return acc/batch_size

def validation(model, validateloader, criterion,tau_values=[0,0,0,0,0,0,0,0,0,0]):
    
    val_loss = 0
    accuracy = 0
    tl=0
    for idx,(images, target) in tqdm(enumerate(validateloader), total=len(validateloader), leave=False):
        # y is the new representation of the label for the image
        # target_trans in line 212 gives the representation of y
        y = target[1]
        # target is the true label of the image
        target = target[0]
        #print(y)
        y = y.to(device)
        images, target = images.to(device), target.to(device)
        output = model.forward(images) 
        val_loss += criterion(output, target).item()
        accuracy+=get_accuracy(output.data,y)
        tl += tLoss(output.data,y,tau_values)
    
    return val_loss, accuracy, tl

def train_classifier():

        epochs = 10
        _epoch = []
        steps = 0
        print_every = 40
        train_error = []
        val_error = []
        for e in range(epochs):
            _epoch.append(e+1)
            model.train()
            running_loss = 0
            train_accuracy = 0
            total_training_samples = 0
            for idx,(images, target) in tqdm(enumerate(train_loader), total=len(train_loader), leave=False):
                steps += 1
                print(images)
                print(target)
                # y is the new representation of the label for the image
                # target_trans in line 212 gives the representation of y
                y = target[1]
                # target is the true label of the image
                target = target[0]
                y = y.to(device)
                images= images.to(device)
                target = target.to(device)
                
                #print(y.shape)
                #print(target.shape)
                #images = torch.autograd.Variable(images, requires_grad=False)
                #target = torch.autograd.Variable(target, requires_grad=False)
                output = model.forward(images)

                loss = criterion(output, target)

                train_accuracy+=get_accuracy(output.data,y)
                     
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                #total_training_samples += target.size(0)
                running_loss += loss.item()
            scheduler.step(running_loss/len(train_loader))
            model.eval()
                  
            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
              validation_loss, accuracy ,tree_loss = validation(model, test_loader, criterion)
            #validation_loss = 0
            #accuracy = 0 
            #tree_loss = 0   
            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/len(train_loader)),
                  "Training Accuracy :{:.3f}.. ".format(train_accuracy/len(train_loader)),
                  "Validation Loss: {:.3f}.. ".format(validation_loss/len(test_loader)),
                  "Validation Accuracy: {:.3f}".format(accuracy/len(test_loader)),
                  "Tree Loss: {:.3f}".format(tree_loss/len(test_loader)))
            train_error.append(running_loss/len(train_loader))
            #val_error.append(validation_loss/len(val_loader))
            running_loss = 0
            if (e+1)%1==0:
              save_model()

#if train_model = True then train the model otherwise skip the training part 
if train_model :
	print("training of the model started")		
	train_classifier()
	print("training of the model finished")

print("evaluation of the model started")
#validating the model
model.eval()
with torch.no_grad():
        validation_loss, accuracy, tree_loss = validation(model, test_loader, criterion)
print("for test data")
print(validation_loss/len(test_loader),accuracy/len(test_loader),tree_loss/len(test_loader))


#this part search for the best tau configuration on validation set
import itertools
tau_values=[0.1,0.2,0.4,0.6,0.9]
min_loss = 1000000
best_tau_config = []
model.eval()
for i in range(len(tau_values)):
  for j in range(len(tau_values)):
    tau = []
    tau.append(tau_values[i])
    diff = (tau_values[j]-tau_values[i])/9
    for k in range(1,9):
      tau.append(tau[k-1]+diff)
    tau.append(tau_values[j])
    print(tau)
    with torch.no_grad():
      loss,accuracy,treeloss = validation(model,val_loader,criterion,tau)
      print("{}/{}".format(i,25),"Test Loss: {:.3f}.. ".format(loss/len(val_loader)),
                  "Test Accuracy :{:.3f}.. ".format(accuracy/len(val_loader)),
                  "Tree Loss: {:.3f}".format(treeloss/len(val_loader)))
      if treeloss<min_loss:
        min_loss = treeloss
        best_tau_config = tau
      print("min loss so far : ",min_loss/len(val_loader))
print(best_tau_config)
print(min_loss/len(val_loader))

#calculating accuracy, treeloss on test data using best_tau_config
with torch.no_grad():
        validation_loss, accuracy, tree_loss = validation(model, test_loader, criterion)
print("for test data")
print(validation_loss/len(test_loader),accuracy/len(test_loader),tree_loss/len(test_loader))
