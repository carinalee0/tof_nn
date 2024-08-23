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
modelmin0_path = "//home//jovyan//simion_files//TOF_ML//src//simulations//modelresults//model_v2_epoch50.pth"
modelmin0 = NN()
modelmin0.load_state_dict(torch.load(modelmin0_path, weights_only=True))

summary_df_path = "//home//jovyan//simion_files//TOF_ML//src//simulations//summary_df_v1"
summary_df = pd.read_pickle(summary_df_path)

try:
    train0_label = torch.tensor(np.array(summary_df['tof_min'][0:1000]), dtype=torch.float)
    train0_data_all = torch.tensor(np.array(summary_df[['initial_ke','mid1_ratio','mid2_ratio','retardation']]), dtype=torch.float)
    train0_data = train0_data_all[0:1000,:]
    # train0_dict = torch.load("//home//jovyan//simion_files//TOF_ML//src//simulations//train_data_v2.pth", weights_only=True)
    # train0_data = train0_dict['data']
    # train0_label = train0_dict['labels']
    train0_SIMdataset = SIMdataset(train0_data, train0_label)
    #print(f'Loaded train_data_v2_2.pth: train data has length {len(trainv2_2_data)}')
    num_cores = 1 #multiprocessing.cpu_count()
    batch_size = 256
    train0_loader = torch.utils.data.DataLoader(train0_SIMdataset, shuffle=True, batch_size=batch_size, num_workers=num_cores)
    #print(f'Got train loader.')
except Exception as e:
    print(e)
trainmin0_output = test_model(modelmin0, train0_loader)
print(f'Output tensor has shape{trainmin0_output.shape}')

plt.scatter(train0_label, trainmin0_output, alpha=0.35, marker='.', color='royalblue', label='train')
# plt.scatter(teststd_label, teststd_output+0.2, alpha=0.35, marker='.', color='mediumseagreen', label='test + 0.2')
# # plt.xlim(0,0.4)
# # plt.ylim(0,0.4)
# plt.axhline(y=0.2, color='r', linestyle=':')
# plt.axhline(y=0, color='r', linestyle=':')
plt.xlabel('Truth tof_min')
plt.ylabel('Predicted tof_min')
plt.title('Predicted vs. Actual tof_min, lr=1e-3, epochs=50')
plt.legend()
plt.show()

# Plot loss
train_loss_tofmin_lr1e3 = pd.read_pickle("/home/jovyan/simion_files/TOF_ML/src/simulations/train_loss_tofmin_lr1e3")
plt.loglog(train_loss_tofmin_lr1e3['loss'])
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.title(' tof_min loss on loglog, lr=1e-3, epochs=50')