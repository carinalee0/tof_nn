import numpy as np
import pandas as pd
import os
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
import multiprocessing
import torch
import torch.utils
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from nn_v2 import SIMdataset, NN, test_model

# Load model
model_path = "//home//jovyan//simion_files//TOF_ML//src//simulations//modelresults//model_v2.pth"
model = NN()
model.load_state_dict(torch.load(model_path, weights_only=True))

# Evaluate against test data
try:
    test_dict = torch.load("//home//jovyan//simion_files//TOF_ML//src//simulations//modelresults//test_data_v2.pth", weights_only=True)
    test_data = test_dict['data']
    test_label = test_dict['labels']
    test_SIMdataset = SIMdataset(test_data, test_label)
    print(f'Loaded test_data_v2.pth: test data has length {len(test_data)}')
    batch_size_test = 256
    num_cores = 1 #multiprocessing.cpu_count()
    test_loader = torch.utils.data.DataLoader(test_SIMdataset, shuffle=True, batch_size=batch_size_test, num_workers=num_cores)
    print(f'Got test loader.')
except Exception as e:
    print(e)

test_output = test_model(model, test_loader) # Each element in output_list is 1 batch.
print(f'Output tensor has shape{output_tensor.shape}')

# Plot results
plt.plot(test_label, output_tensor)
plt.xlabel('Truth tof_min')
plt.ylabel('Predicted tof_min')
plt.show()

# Evaluate against training data
# Load test data
try:
    train_dict = torch.load("//home//jovyan//simion_files//TOF_ML//src//simulations//modelresults//train_data_v2.pth", weights_only=True)
    train_data = train_dict['data']
    train_label = train_dict['labels']
    train_SIMdataset = SIMdataset(train_data, train_label)
    print(f'Loaded train_data_v2.pth: train data has length {len(train_data)}')
    batch_size_test = 256
    num_cores = 1 #multiprocessing.cpu_count()
    train_loader = torch.utils.data.DataLoader(train_SIMdataset, shuffle=True, batch_size=batch_size_test, num_workers=num_cores)
    print(f'Got train loader.')
except Exception as e:
    print(e)

train_output = test_model(model, train_loader)
print(f'Output tensor has shape{train_output.shape}')




