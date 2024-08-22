"""
Prepping data from MRCO SIMION database for model training.
Combines h5 files from all simulations, converts to 1 datafame of inputs and outputs.

"""
import h5py
import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
from scipy.stats import norm
import re
import tablesS
import seaborn as sns
sns.set_theme('notebook')

# 1: Combine all h5 files into 1
database = os.listdir('C:\\Users\\carina\\simion_files\\TOF_ML\\simulations\\TOF_simulation_data_Vandana')
h5_list = []
for folder in database:
    folder_path = os.listdir(os.path.join('C:\\Users\\carina\\simion_files\\TOF_ML\\simulations\\TOF_simulation_data_Vandana', folder))
    #print(folder_path)
    for file in folder_path:
        if '.h5' in file:
            h5_list.append(os.path.join('C:\\Users\\carina\\simion_files\\TOF_ML\\simulations\\TOF_simulation_data_Vandana', folder, file))
print(f'Length of h5 list: {len(h5_list)}')

def float_match(match):
    if 'R' in match:
        match = match.replace('R','')
    valueMatch = float(match.replace('neg','-').replace('pos','').replace('_',''))
    return valueMatch 

def getVoltages(fileName):
    splitName = fileName.split('_')
    if 'sim' in splitName[5]:
        retardationMatch = float_match(splitName[6] + splitName[7])
        mid1Match = float_match(splitName[8] + splitName[9])
        mid2Match = float_match(splitName[10] + splitName[11])
    else: 
        print('Reindex split Name?')
    return retardationMatch, mid1Match, mid2Match

df_list = []
for file in h5_list:
    try:
        with h5py.File(file, 'r+') as f:
        # 2: Add voltages as keys
            try:
                retardationValue, mid1Value, mid2Value = getVoltages(file)
                f['data1'].create_dataset(name='retardation', data=retardationValue)
                f['data1'].create_dataset(name='mid1_ratio', data=mid1Value)
                f['data1'].create_dataset(name='mid2_ratio', data=mid2Value)
            except:
                pass
                #print(f'Datasets already exist in {file}.')
                
            # 3: Convert h5 file to dataframe
            df_from_h5 = pd.DataFrame()
            for key in f['data1'].keys():
                array = np.array(f['data1'][key][()])
                df_from_h5[key] = array
            if df_from_h5.shape[0]  == 0:
                print(file)   
            df_list.append(df_from_h5)
    except Exception as e:
        print(f'Adding voltags as keys exception: {e}')
        
# 4: Select inputs and outputs   
collectedParticles_df_list=[]
empty_dfs = 0
for df in df_list:
    collectedParticles_df = (
    df
    .query('x > 403.0')
    .groupby(['initial_ke', 'mid1_ratio', 'mid2_ratio', 'retardation'], as_index = False)
    .agg(tof_min=('tof', 'min'), tof_std=('tof','std'))
    .dropna()
    )
    if collectedParticles_df.shape[0] > 0:
        collectedParticles_df_list.append(collectedParticles_df)
    else:
        empty_dfs +=1

print(f'Length of collectedParticles df: {len(collectedParticles_df_list)}')
print(f'Empty data frames: {empty_dfs}')
#print(len(collectedParticles_df_list)+empty_dfs)

# Put inputs and outputs into 1 data frame.
summary_df=pd.concat(collectedParticles_df_list)
print(summary_df.describe())