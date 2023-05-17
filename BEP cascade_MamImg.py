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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ",device)

#speed up things a little bit
#cudnn.benchmark = True
# flag to train the model   if train_model = true then train the model else test the model
train_model = True

#contains the location of the saved model or model to be saved
model_location = "/savedmodels/extracted_short_imagenet/checkpoints/bep(1layer).pth.tar"

#path to directories containing train and test data
data_dir = r'/savedmodels/extracted_short_imagenet'
train_dir = data_dir + '/train'
test_dir = data_dir + '/val'
print("train_dir : ",train_dir)
print("test_dir : ",test_dir)

nodesAtLevel = []*10
nodesAtLevel.append(1)
nodesAtLevel.append(4)
nodesAtLevel.append(10)
nodesAtLevel.append(29)
nodesAtLevel.append(46)
nodesAtLevel.append(64)
nodesAtLevel.append(50)
nodesAtLevel.append(75)
nodesAtLevel.append(40)
nodesAtLevel.append(4)

logNodesAtLevel = []*10
logNodesAtLevel.append(1)
logNodesAtLevel.append(3)
logNodesAtLevel.append(4)
logNodesAtLevel.append(5)
logNodesAtLevel.append(6)
logNodesAtLevel.append(7)
logNodesAtLevel.append(6)
logNodesAtLevel.append(7)
logNodesAtLevel.append(6)
logNodesAtLevel.append(8)


# function to calculate tree loss using BEP cascade prediction 
def tLoss(output,target,tau_values=[0,0,0,0,0,0,0,0,0,0]):
  batch_size = target.size(0)
  num_classes = output.size(1)
  h = 9
  val = None
  pred = None
  loss = 0
  global labels
  global level
  l1 = logNodesAtLevel[0]
  l2 = logNodesAtLevel[1]
  l3 = logNodesAtLevel[2]
  l4 = logNodesAtLevel[3]
  l5 = logNodesAtLevel[4]
  l6 = logNodesAtLevel[5]
  l7 = logNodesAtLevel[6]
  l8 = logNodesAtLevel[7]
  l9 = logNodesAtLevel[8]
  l10 = logNodesAtLevel[9]
  # Loop for each example in the batch
  for i in range(batch_size):
    # t is the actual target of the ith example in the batch
    t = target[i]
    # Go searching bottom-up in the hierarchy tree
    h = 9
    while h >= 0:
      # Clone the output array each time for checking at every level in the bottom-up search 
      values = output.clone().detach()
      # At each level in the bottom-up traversal, we take the minimum of the outputs of nodes
      # (a particular range in the totat set of output nodes of the model) and if it is 
      # greater than the threshold at that level, we break and get class index at level from 
      # the signs of those outputs (btoi function defined above)
      if h == 0:
        values = values[i,0:l1]
        val,_ = torch.min(torch.abs(values),0)        
        if val >= tau_values[0]:
          pred = b2i(values,h)
          pred_label = level[h][pred]
          break
      if h == 1:
        values = values[i,l1:l1+l2]
        val,_ = torch.min(torch.abs(values),0)        
        if val >= tau_values[1]:
          pred = b2i(values,h)
          pred_label = level[h][pred]
          break
      if h == 2:
        values = values[i,l1+l2:l1+l2+l3]
        val,_ = torch.min(torch.abs(values),0)        
        if val >= tau_values[2]:
          pred = b2i(values,h)
          pred_label = level[h][pred]
          break
      if h == 3:
        values = values[i,l1+l2+l3:l1+l2+l3+l4]
        val,_ = torch.min(torch.abs(values),0)        
        if val >= tau_values[3]:
          pred = b2i(values,h)
          pred_label = level[h][pred]
          break
      if h == 4:
        values = values[i,l1+l2+l3+l4:l1+l2+l3+l4+l5]
        val,_ = torch.min(torch.abs(values),0)        
        if val >= tau_values[4]:
          pred = b2i(values,h)
          pred_label = level[h][pred]
          break
      if h == 5:
        values = values[i,l1+l2+l3+l4+l5:l1+l2+l3+l4+l5+l6]
        val,_ = torch.min(torch.abs(values),0)        
        if val >= tau_values[h]:
          pred = b2i(values,h)
          pred_label = level[h][pred]
          break
      if h == 6:
        values = values[i,l1+l2+l3+l4+l5+l6:l1+l2+l3+l4+l5+l6+l7]
        val,_ = torch.min(torch.abs(values),0)        
        if val >= tau_values[h]:
          pred = b2i(values,h)
          pred_label = level[h][pred]
          break
      if h == 7:
        values = values[i,l1+l2+l3+l4+l5+l6+l7:l1+l2+l3+l4+l5+l6+l7+l8]
        val,_ = torch.min(torch.abs(values),0)        
        if val >= tau_values[h]:
          pred = b2i(values,h)
          pred_label = level[h][pred]
          break
      if h == 8:
        values = values[i,l1+l2+l3+l4+l5+l6+l7+l8:l1+l2+l3+l4+l5+l6+l7+l8+l9]
        val,_ = torch.min(torch.abs(values),0)        
        if val >= tau_values[h]:
          pred = b2i(values,h)
          pred_label = level[h][pred]
          break
      if h == 9:
        values = values[i,l1+l2+l3+l4+l5+l6+l7+l8+l9:l1+l2+l3+l4+l5+l6+l7+l8+l9+l10]
        val,_ = torch.min(torch.abs(values),0)        
        if val >= tau_values[h]:
          pred = b2i(values,h)
          pred_label = labels[pred]
          break
            
      h = h - 1

    # tree_loss() function gives the tree distance between the two nodes
    # Total loss for the batch is calculated
    loss = loss + tree_loss(labels[t],pred_label)

  loss = loss/batch_size

  # Average loss for the batch is returned
  return loss

# function to convert integer to binary     target is integer to be converted
def itob(target,bits):
  encoding=-np.ones(bits)
  ''' -1 in place of 0 in binary represantation of a number'''
  j = bits-1
  while(target!=0):
    if (target%2)==1 :
      encoding[j] = 1
    target = target//2
    j = j-1
  return encoding

#function to convert a batch of binary numbers into integer
def btoi(binary,h):
  # this is to convert a -1 and 1 to 0 and 1 respectively
  batch_size = binary.size()[0]
  binary = -torch.sign(binary)
  bits = logNodesAtLevel[h]
  binary = (binary+1)/2
  #print(binary.shape)
  j = bits-1
  nodes = [1,4,10,29,46,64,50,75,40,218]
  j = bits-1 
  decoded_target = torch.zeros(batch_size).to(device)
  while j>=0:
    decoded_target = decoded_target + binary[:,j] * (2**(bits-j-1))
    j = j-1
  
  return decoded_target

#function to convert a single binary number into integer
def b2i(binary,h):
  batch_size = binary.size()[0]
  binary = -torch.sign(binary)
  bits = logNodesAtLevel[h]
  binary = (binary+1)/2
  #print(binary.shape)
  j = bits-1
  nodes = [1,4,10,29,46,64,50,75,40,218]

  max_label = nodes[h]
  decoded_target = 0
  while j>=0:
    decoded_target = decoded_target + binary[j] * (2**(bits-j-1))
    j = j-1
  if decoded_target >= max_label:
    decoded_target = decoded_target - 2**(bits-1)
  return decoded_target.type(torch.LongTensor)

#function to translate the target labels according to the model
def target_translate(target):
    translated_target = []  # target is translated into 19 bits (1+1+4+6+7)
    # 5 is the height of the tree
    target_label = labels[target]
    for i in range(9):
      ancestor = get_ancestor(target_label,i)
      if ancestor==target_label and (ancestor not in level[i]):
        translated_target.extend(itob(nodesAtLevel[i]+1,logNodesAtLevel[i]))
      else:
        translated_target.extend(itob(level[i].index(ancestor),logNodesAtLevel[i]))
    translated_target.extend(itob(target,logNodesAtLevel[9]))
    translated_target = torch.FloatTensor(translated_target)
    return target, translated_target   




class SoloDataset(Dataset):
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
        y_label = (int(self.annotations.iloc[index,1]),torch.tensor(self.file[int(self.annotations.iloc[index,1])],dtype=torch.int64))
        return (x_data.float(), y_label)

num_classes = 53
learning_rate = 0.000001
batch_size = 16
num_epochs = 30
input_size = 2048

# Load Data

train_data = SoloDataset(
    csv_file="data/extracted_short_imagenet/train_labels.csv",
    path1="data/extracted_short_imagenet/train",
    path2="data/extracted_short_imagenet/job/target_labelsbepold.npy",
    transform=transforms.ToTensor(),
    target_transform = target_translate)

test_data = SoloDataset(
    csv_file="data/extracted_short_imagenet/val_labels.csv",
    path1="data/extracted_short_imagenet/val",
    path2="data/extracted_short_imagenet/job/target_labelsbepold.npy",
    transform=transforms.ToTensor(),
    target_transform=target_translate)
    
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
        self.fc1 = nn.Linear(input_size, num_classes)
        self.relu = nn.ReLU()
        #self.fc2 = nn.Linear(512,128)
        #self.fc3 = nn.Linear(128,num_classes)
    def forward(self, x):
        x = self.relu(x)
        x = self.fc1(x)
        #x = self.relu(x)
        #x = self.fc2(x)
        #x = self.relu(x)
        #x = self.fc3(x)
        return x


#loss function
class BEP_Loss(torch.nn.Module):
  def __init__(self):
        super(BEP_Loss,self).__init__()
  def forward(self,x,y):
    # shape of x and y is batch_size * 19
    batch_size = x.size()[0]
    # shape of z is batch_size * 19
    z = x*y
    start = 0
    #shape of zeros is batch_size
    loss = 0 
    zeros = torch.zeros(batch_size).to(device)
    #loop for each level
    for i in range(10):
      t1 = z[:,start:start+logNodesAtLevel[i]]
      w,_ = torch.max(t1,1)
      w1 = torch.max(w+1,zeros)
      loss = loss + torch.sum(w1)
      start = start+logNodesAtLevel[i]

    loss = loss/batch_size
    return loss

# Model
model = NN(input_size=input_size, num_classes=num_classes).to(device)

# Loss and optimizer
criterion = BEP_Loss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
#optimizer = torch.optim.Adadelta(model.parameters(), lr=0.01, rho=0.9, eps=1e-3, weight_decay=0.001)
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
  pred=btoi(output[:,45:53],9)
  correct = pred==target
  #print(correct)
  return correct.float().sum()/batch_size
  
def validation(model, validateloader, criterion,tau_values=[0,0,0,0,0,0,0,0,0,0]):
    
    val_loss = 0
    accuracy = 0
    tl=0
    for idx,(images, target) in tqdm(enumerate(validateloader), total=len(validateloader), leave=False):
        # target here conatains two things [actual_target, translated_target]
        #translated target is the new representation of the label for the image
        translated_target = target[1]
        # target is the true label of the image
        target = target[0]
        translated_target = translated_target.to(device)
        images, target = images.to(device), target.to(device)
        output = model.forward(images) 
        val_loss += criterion(output, translated_target).item()
        accuracy+=get_accuracy(output.data,target)
        tl += tLoss(output.data,target,tau_values)
    
    return val_loss, accuracy, tl

def train_classifier():

        #for parameter in model.parameters():
        #    parameter.requires_grad = True
        epochs = 30
        _epoch = []
        steps = 0
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
                # target here conatains two things [actual_target, translated_target(19 bits)]
                #translated target is the new representation of the label for the image
                translated_target = target[1]
                # target is the true label of the image
                target = target[0]
                translated_target = translated_target.to(device)
                images= images.to(device)
                target = target.to(device)
                
                #print(y.shape)
                #print(target.shape)
                images = torch.autograd.Variable(images, requires_grad=False)
                target = torch.autograd.Variable(target, requires_grad=False)
                translated_target = torch.autograd.Variable(translated_target, requires_grad=False)
                output = model.forward(images)
                loss = criterion(output, translated_target)

                train_accuracy+=get_accuracy(output.data,target)
                     
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                #total_training_samples += target.size(0)
                running_loss += loss.item()
            scheduler.step(running_loss/len(train_loader))
                  
            validation_loss = 0
            accuracy = 0 
            tree_loss = 0
            if e%1==0:
              model.eval()
              # Turn off gradients for validation, saves memory and computations
              with torch.no_grad():
                validation_loss, accuracy ,tree_loss = validation(model, test_loader, criterion)   
              print("Epoch: {}/{}.. ".format(e+1, epochs),
                    "Training Loss: {:.3f}.. ".format(running_loss/len(train_loader)),
                    "Training Accuracy :{:.3f}.. ".format(train_accuracy/len(train_loader)),
                    "Validation Loss: {:.3f}.. ".format(validation_loss/len(test_loader)),
                    "Validation Accuracy: {:.3f}".format(accuracy/len(test_loader)),
                    "Tree Loss: {:.3f}".format(tree_loss/len(test_loader)))
            
            train_error.append(running_loss/len(train_loader))
            #val_error.append(validation_loss/len(val_loader))
            running_loss = 0
            # save model after every 2 epochs
            if (e+1)%2==0:
              save_model()

#if train_model = True then train the model otherwise skip the training part 
if train_model :
	print("training of the model started")		
	train_classifier()
	print("training of the model finished")

model.eval()
with torch.no_grad():
    validation_loss, accuracy, treeloss = validation(model, test_loader, criterion)

print(validation_loss/len(test_loader),accuracy/len(test_loader),treeloss/len(test_loader))

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
    validation_loss, accuracy, treeloss = validation(model, test_loader, criterion,best_tau_config)

print(validation_loss/len(test_loader),accuracy/len(test_loader),treeloss/len(test_loader))
