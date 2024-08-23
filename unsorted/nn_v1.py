import numpy as np
import pandas as pd
import os
import seaborn as sns
import pickle
import matplotlib.pyplot as plt

import multiprocessing
from sklearn.model_selection import train_test_split

import torch
import torch.utils
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)
cwd = os.getcwd()

normalized_data_path = "//home//jovyan//simion_files//TOF_ML//simulations//TOF_data//08082024_normalized"
normalized_data_raw = pd.read_pickle(normalized_data_path)
# <KeysViewHDF5 ['final_azimuth', 'final_elevation', 'final_ke', 'initial_azimuth', 'initial_elevation', 'initial_ke', 'ion_number', 'tof', 'x', 'y', 'z']>
# the 4 inputs are initial ke, ke fwhm, blade 22, blade 26
# outputs are tof, tof fwhm
#print(normalized_data_raw.shape)

# Flatten data
normalized_data_tensor = torch.tensor(normalized_data_raw, dtype= torch.float)
normalized_data_tensor = normalized_data_tensor.permute(0, 2, 1).flatten(start_dim=0, end_dim=1)
print(normalized_data_tensor.shape)

# Redirect data so that PyTorch is not working with directly
class SIMdataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label

labels = normalized_data_tensor[:,7] # Need tof fwhm 
data = normalized_data_tensor[:,5] # Initial KE from the h5 file. TODO: also need to include blade voltages and fwhm

# Split training and test data. Shuffles data.
data_train, data_test, label_train, label_test = train_test_split(
    data, 
    labels, 
    test_size=0.25, 
    shuffle=True,  
    random_state=random_seed
)

# Dataloader
train_data = SIMdataset(data_train, label_train)
test_data = SIMdataset(data_test, label_test)

num_cores = multiprocessing.cpu_count()
train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, num_workers=num_cores)
test_loader = torch.utils.data.DataLoader(test_data, shuffle=True, num_workers=num_cores)

# Define a class for the NN
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        
        hidden_size = 20
        
        self.linear1 = nn.Linear(10, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        
        #propagate through model
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        
        return torch.linear(x)

def train_model(model, epochs, train_loader, optimizer):
    
    train_losses = []
    train_counter = []
    
    #set network to training mode
    model.train()
    for epoch in range(epochs):
        #iterate through data batches
        for batch_idx, (data, target) in enumerate(train_loader):

            #reset gradients
            optimizer.zero_grad()

            #evaluate network with data
            output = model(data)

            #compute loss and derivative
            mseLoss = nn.MSELoss()
            loss = mseLoss(output, target) # target is label
            loss.backward()

            #step optimizer
            optimizer.step()

            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            torch.save(model.state_dict(), cwd + '/modelresults/model.pth')
            torch.save(optimizer.state_dict(), cwd + '/modelresults/optimizer.pth')
        
    return train_losses, train_counter

def test_model(model,loader):
    test_losses = []
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            output = model(data)
            mseLoss = nn.MSELoss()
            test_loss += mseLoss(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    return test_losses

fully_connected_model = NN()
optimizer = optim.SGD(fully_connected_model.parameters())
log_interval = 2

#train_model(fully_connected_model, 5, train_loader, optimizer)
test_model(fully_connected_model, test_loader)