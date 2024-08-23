"""
8/19/2024 version 4
batch size = 100
learning rate = default?
optimizer = Adam
epochs = 25
hidden size = 20
layers = 2

"""
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

summary_df_path = "//home//jovyan//simion_files//TOF_ML//src//simulations//summary_df_v1"
summary_df = pd.read_pickle(summary_df_path)
# The 4 inputs are initial ke, ke fwhm, blade 22, blade 26. Outputs are tof, tof fwhm

# Redirect data so that PyTorch is not working with directly
class SIMdataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        try:
            sample = self.data[idx]
            label = self.labels[idx]
        except:
            pass
        return sample, label

tof_min_label = torch.tensor(np.array(summary_df['tof_min']), dtype=torch.float)
#Change inputs to tensor
data = torch.tensor(np.array(summary_df[['initial_ke','retardation','mid1_ratio','mid2_ratio']]), dtype=torch.float)

# Split training and test data. Shuffles data.
data_train, data_test, label_train, label_test = train_test_split(
    data, 
    tof_min_label, 
    test_size=0.25, 
    shuffle=True,  
    random_state=random_seed
)

# Dataloader
train_data = SIMdataset(data_train, label_train)
test_data = SIMdataset(data_test, label_test)

num_cores = multiprocessing.cpu_count()
# batch sizes reduced from 256 in v2
batch_size_train=100
batch_size_test=100
train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size_train, num_workers=num_cores)
test_loader = torch.utils.data.DataLoader(test_data, shuffle=True, batch_size=batch_size_test, num_workers=num_cores)
# Save the data and labels
torch.save({
    'data': data_train,
    'labels': label_train
}, 'train_data_v4.pth')

torch.save({
    'data': data_test,
    'labels': label_test
}, 'test_data_v4.pth')

# Define a class for the NN
# v3: added one more relu 
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        
        hidden_size = 20
        
        self.linear1 = nn.Linear(4, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        
        #propagate through model
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        
        return x

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
            torch.save(model.state_dict(), cwd + '/modelresults/model_v4.pth')
            torch.save(optimizer.state_dict(), cwd + '/modelresults/optimizer_v4.pth')
        
    return train_losses, train_counter


def test_model(model,loader):
    model.eval()
    output_tensor = torch.empty(0, 1)
    with torch.no_grad():
        #for data, target in loader:
        for batch_idx, (data, target) in enumerate(loader):
            output = model(data)
            output_tensor = torch.cat((output_tensor, output), 0)
            
    return output_tensor

fully_connected_model = NN()
optimizer = optim.Adam(fully_connected_model.parameters())
log_interval = 2

if __name__ == "__main__":
    train_model(fully_connected_model, 25, train_loader, optimizer)
    test_model(fully_connected_model, test_loader)
