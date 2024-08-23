"""
8/20/2024---only saving the data here.
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
data = torch.tensor(np.array(summary_df[['initial_ke','mid1_ratio','mid2_ratio', 'retardation']]), dtype=torch.float)

# Split training and test data. Shuffles data.
X_train, X_test, label_train, label_test = train_test_split(
    data, 
    tof_min_label, 
    test_size=0.25, 
    shuffle=True,  
    random_state=random_seed
)

# Save the data and labels
torch.save({
    'data': X_train,
    'labels': label_train
}, 'train_tofmin.pth')
print('Saved training dataset.')

torch.save({
    'data': X_test,
    'labels': label_test
}, 'test_tofmin.pth')
print('Saved test dataset.')