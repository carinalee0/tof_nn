"""
8/20/2024--- pulling test and train data from saved_testandtrain.py
concerned that indices are getting mixed up. look up how to make a np array into a tensor.
batch size = 256
learning rate = 1e-3
optimizer = Adam
epochs = 50
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

# 8/20: store as a deep copy, train dataset, save as a global variable
train_dict = torch.load('/home/jovyan/simion_files/TOF_ML/src/simulations/train_tofmin.pth', weights_only=True)
X_train = train_dict['data']
label_train = train_dict['labels']
train_dataset = SIMdataset(X_train, label_train)
# test_dict = torch.load('/home/jovyan/simion_files/TOF_ML/src/simulations/test_tofmin.pth', weights_only=True)
# X_test = train_dict['data']
# label_test = train_dict['labels']
# test_dataset = SIMdataset(X_test, label_test)
num_cores = multiprocessing.cpu_count()
batch_size_train=256
batch_size_test=256
train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=False, batch_size=batch_size_train, num_workers=num_cores)
# test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=True, batch_size=batch_size_test, num_workers=num_cores)
# 8/20: shuffling twice should be okay -ryan

# Define a class for the NN
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
        for batch_idx, (input_data, target) in enumerate(train_loader):

            #reset gradients
            optimizer.zero_grad()

            #evaluate network with data
            output = model(input_data)

            #compute loss and derivative
            mseLoss = nn.MSELoss()
            loss = mseLoss(output, target) # target is label
            loss.backward()

            #step optimizer
            optimizer.step()

            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(input_data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))
                train_losses.append(loss.item())
    
    train_loss_df = pd.DataFrame(train_losses, columns=['loss'])
    train_loss_df.to_pickle('train_loss_tofmin_lr1e3_noLoaderShuffle')
    torch.save(model.state_dict(), cwd + '/modelresults/model_tofmin_lr1e3_noLoaderShuffle.pth')
    torch.save(optimizer.state_dict(), cwd + '/modelresults/optimizer.pth')
    
    # 8/20: for train_model, return the model. then model.eval(). outside of training loop, do it as training data and plot it. 
    model.eval()
    train_output=model(X_train[-94:-1])
    plt.scatter(target.detach().numpy(), train_output.detach().numpy(), marker='.')
    plt.xlabel('True')
    plt.ylabel('Predicted')
    plt.savefig('temp_noLoaderShuffle.png')
    plt.show()
        
    return model, train_losses

# def test_model(model,loader):
#     test_losses = []
#     model.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for data, target in loader:
#             output = model(data)
#             mseLoss = nn.MSELoss()
#             test_loss += mseLoss(output, target).item()
#             pred = output.data.max(1, keepdim=True)[1]
#             correct += pred.eq(target.data.view_as(pred)).sum()
#     test_loss /= len(test_loader.dataset)
#     test_losses.append(test_loss)
#     print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         test_loss, correct, len(test_loader.dataset),
#         100. * correct / len(test_loader.dataset)))
    
#     return test_losses

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
optimizer = optim.Adam(fully_connected_model.parameters(), lr=0.001)
log_interval = 5

if __name__ == "__main__":
    train_model(fully_connected_model, 50, train_loader, optimizer)
    test_model(fully_connected_model, train_loader)
