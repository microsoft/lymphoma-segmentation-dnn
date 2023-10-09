#%% 
'''
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
'''
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import os 
import sys 
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
sys.path.append(PARENT_DIR)
from config import RESULTS_FOLDER
# %%
device_rank = 2
fold = 0
network = ['unet', 'segresnet', 'dynunet', 'swinunetr']
p = 2
extra_features = f'p{p}_wd1em5_lr2em4'
N_for_network = {
    'unet': 224,
    'segresnet': 192,
    'dynunet': 160,
    'swinunetr': 128
}

inputsize = [N_for_network[n] for n in network]
experiment_code = [f"{network[i]}_fold{fold}_randcrop{inputsize[i]}_{extra_features}" for i in range(len(network))]
valid_or_test = 'test'
#%%
save_testmetrics_dir = os.path.join(RESULTS_FOLDER, f'{valid_or_test}_metrics')
save_testmetrics_dir = [os.path.join(save_testmetrics_dir, 'fold'+str(fold), network[i], experiment_code[i]) for i in range(len(network))]
save_testmetrics_fname = [os.path.join(dir, f'{valid_or_test}metrics.csv') for dir in save_testmetrics_dir]

# %%
dfs = [pd.read_csv(fname) for fname in save_testmetrics_fname]
# %%
colors = np.random.rand(88, 3)
def plot_detection_sensitivity_fp_per_patient(df, method='method1', ax=None, rand_colors = colors):
    if method == 'method1':
        suffix = 'M1'
    elif method == 'method2':
        suffix = 'M2'
    elif method == 'method3':
        suffix = 'M3'
    else:
        suffix = None
    
    tp = np.array(df[f'TP_{suffix}'].astype(int).values)
    fp = np.array(df[f'FP_{suffix}'].astype(int).values)
    fn = np.array(df[f'FN_{suffix}'].astype(int).values)
    
    sensitivity = tp/(tp + fn)
    fdr = fp/(tp + fp)
    ax.scatter(fdr, sensitivity, c=rand_colors, marker='o', s=200, alpha=0.5)
    for i, (xi, yi) in enumerate(zip(fp, sensitivity)):
        ax.annotate(str(i+1), (xi, yi), ha='center', va='center', fontsize=8)

    print(round(np.mean(sensitivity), 2))
    print(round(np.median(sensitivity), 2))
    print(round(np.mean(fdr), 2))
    print(round(np.median(fdr), 2))
    return ax
#%%
fig, ax = plt.subplots(4, 3, figsize=(15,20))
fig.patch.set_facecolor('white')
fig.patch.set_alpha(1)
colors = ['red', 'blue', 'green']
networks = ['UNet', 'SegResNet', 'DynUNet', 'SwinUNETR']
xlims_values = [(-0.5, 200), (-0.5, 125), (-0.5, 120)]
for i in range(len(networks)):
    for j in range(len(colors)):
        axs = ax[i][j]
        axs.grid(True, linestyle='dotted')
        print(f'Network: {networks[i]}; Method: {j+1}')
        plot_detection_sensitivity_fp_per_patient(dfs[i], method=f'method{j+1}', ax=axs)
        if i == 0:
            axs.set_title(f'Method {j+1}', fontsize=25)
        if j == 0:
            axs.set_ylabel(f'{networks[i]}\nDetection sensitivity', fontsize=20)
        if i == len(networks)-1:
            axs.set_xlabel('FP', fontsize=20)
        # axs.set_xlim(xlims_values[j])
        # axs.set_xscale('log')
        
#%%
def plot_detection_sensitivity_vs_fp_histograms(df, method='method1', ax=None):
    if method == 'method1':
        suffix = 'M1'
    elif method == 'method2':
        suffix = 'M2'
    elif method == 'method3':
        suffix = 'M3'
    else:
        suffix = None
    
    tp = np.array(df[f'TP_{suffix}'].astype(int).values)
    fp = np.array(df[f'FP_{suffix}'].astype(int).values)
    fn = np.array(df[f'FN_{suffix}'].astype(int).values)

    sensitivity = tp / (tp + fn)
    bin_width = 2
    # Define bins for FP value
    fp_bins = np.arange(1, np.max(fp) + bin_width, bin_width)
    
    # Calculate indices of FP bins for each FP value
    fp_bin_indices = np.digitize(fp, fp_bins) - 1
    
    # Initialize lists to store sensitivity values for each bin
    sensitivity_bins = [[] for _ in range(len(fp_bins))]
    
    # Populate sensitivity values in respective bins
    for idx, sens in enumerate(sensitivity):
        if 0 <= fp_bin_indices[idx] < len(fp_bins):
            sensitivity_bins[fp_bin_indices[idx]].append(sens)
    
    # Calculate mean sensitivity for each bin
    mean_sensitivity_bins = [np.mean(sens_list) for sens_list in sensitivity_bins]
    
    # Plot histograms
    ax.bar(fp_bins, mean_sensitivity_bins, width=bin_width, align='edge')
    return ax


#%%
fig, ax = plt.subplots(4, 3, figsize=(15,20))
fig.patch.set_facecolor('white')
fig.patch.set_alpha(1)
colors = ['red', 'blue', 'green']
networks = ['UNet', 'SegResNet', 'DynUNet', 'SwinUNETR']
for i in range(len(networks)):
    for j in range(len(colors)):
        axs = ax[i][j]
        axs.grid(True, linestyle='dotted')
        print(f'Network: {networks[i]}; Method: {j+1}')
        plot_detection_sensitivity_vs_fp_histograms(dfs[i], method=f'method{j+1}', ax=axs)
        if i == 0:
            axs.set_title(f'Method {j+1}', fontsize=25)
        if j == 0:
            axs.set_ylabel(f'{networks[i]}\nMean detection sensitivity', fontsize=20)
        if i == len(networks)-1:
            axs.set_xlabel('FP', fontsize=20)






# %%
def plot_detection_sensitivity_vs_fdr_histograms(df, method='method1', ax=None):
    if method == 'method1':
        suffix = 'M1'
    elif method == 'method2':
        suffix = 'M2'
    elif method == 'method3':
        suffix = 'M3'
    else:
        suffix = None
    
    tp = np.array(df[f'TP_{suffix}'].astype(int).values)
    fp = np.array(df[f'FP_{suffix}'].astype(int).values)
    fn = np.array(df[f'FN_{suffix}'].astype(int).values)

    sensitivity = tp / (tp + fn + fp)

    fdr = fp / (tp + fp)
    bin_width = 0.05
    # Define bins for FP value
    fdr_bins = np.arange(0, 1 + bin_width, bin_width)
    
    # Calculate indices of FP bins for each FP value
    fdr_bin_indices = np.digitize(fdr, fdr_bins) - 1
    
    # Initialize lists to store sensitivity values for each bin
    sensitivity_bins = [[] for _ in range(len(fdr_bins))]
    
    # Populate sensitivity values in respective bins
    for idx, sens in enumerate(sensitivity):
        if 0 <= fdr_bin_indices[idx] < len(fdr_bins):
            sensitivity_bins[fdr_bin_indices[idx]].append(sens)
    
    # Calculate mean sensitivity for each bin
    mean_sensitivity_bins = [np.mean(sens_list) for sens_list in sensitivity_bins]
    
    # Plot histograms
    ax.bar(fdr_bins, mean_sensitivity_bins, width=bin_width, align='edge')
    return ax

#%%
fig, ax = plt.subplots(4, 3, figsize=(15,20))
fig.patch.set_facecolor('white')
fig.patch.set_alpha(1)
colors = ['red', 'blue', 'green']
networks = ['UNet', 'SegResNet', 'DynUNet', 'SwinUNETR']
for i in range(len(networks)):
    for j in range(len(colors)):
        axs = ax[i][j]
        axs.grid(True, linestyle='dotted')
        print(f'Network: {networks[i]}; Method: {j+1}')
        plot_detection_sensitivity_vs_fdr_histograms(dfs[i], method=f'method{j+1}', ax=axs)
        if i == 0:
            axs.set_title(f'Method {j+1}', fontsize=25)
        if j == 0:
            axs.set_ylabel(f'{networks[i]}\nMean detection sensitivity', fontsize=20)
        if i == len(networks)-1:
            axs.set_xlabel('FP', fontsize=20)
# %%
