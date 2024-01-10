'''
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
'''
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np 
import os 
import glob
import sys
config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(config_dir)
from config import RESULTS_FOLDER

# %%
def plot_train_logs(train_fpaths, valid_fpaths, network_names):
    train_dfs = [pd.read_csv(path) for path in train_fpaths]
    valid_dfs = [pd.read_csv(path) for path in valid_fpaths]

    train_losses = [df['Loss'].values for df in train_dfs]
    valid_metrics = [df['Metric'].values for df in valid_dfs]
    train_epochs = [np.arange(len(train_loss))+1 for train_loss in train_losses]
    valid_epochs = [2*(np.arange(len(valid_metric))+1) for valid_metric in valid_metrics]
    min_losses = [np.min(train_loss) for train_loss in train_losses]
    min_losses_epoch = [np.argmin(train_loss) + 1 for train_loss in train_losses]
    max_dscs = [np.max(valid_metric) for valid_metric in valid_metrics]
    max_dscs_epoch = [2*(np.argmax(valid_metric)+1) for valid_metric in valid_metrics]
    fig, ax = plt.subplots(1,2, figsize=(20,10))
    fig.patch.set_facecolor('white')
    fig.patch.set_alpha(1)

    for i in range(len(train_losses)):
        ax[0].plot(train_epochs[i], train_losses[i])
        ax[1].plot(valid_epochs[i], valid_metrics[i])
        ax[0].plot(min_losses_epoch[i], min_losses[i], '-o', color='red')
        ax[1].plot(max_dscs_epoch[i], max_dscs[i], '-o', color='red')
        
        ax[0].text(np.min(train_epochs[i]), np.min(train_losses[i]), f'Total epochs: {len(train_epochs[i])}', fontsize=15)
        
    legend_labels_trainloss = [f"{network_names[i]}; Min loss: {round(min_losses[i], 4)} ({len(train_epochs[i])})" for i in range(len(network_names))]
    legend_labels_validdice = [f"{network_names[i]}; Max DSC: {round(max_dscs[i], 4)} ({len(valid_epochs[i])})" for i in range(len(network_names))]

    ax[0].legend(legend_labels_trainloss, fontsize=16)
    ax[1].legend(legend_labels_validdice, fontsize=16)
    ax[0].set_title('Train loss', fontsize=25)
    ax[1].set_title('Valid DSC', fontsize=25)
    ax[0].set_ylabel('Dice loss', fontsize=20)
    ax[1].set_ylabel('Dice score', fontsize=20)
    ax[0].grid(True)
    ax[1].grid(True)
    plt.show()

#%%
fold = 0
network = ['unet']
inputsize = [192, 192, 160, 128]
p = 2
inputsize_dict = {
    'unet': 192,
    'attentionunet': 192,
    'segresnet': 192,
    'dynunet': 160,
    'unetr': 160,
    'swinunetr': 128
}

experiment_code = [f"{network[i]}_fold{fold}_randcrop{inputsize[i]}" for i in range(len(network))]
save_logs_dir = os.path.join(RESULTS_FOLDER, 'logs')
save_logs_folders = [os.path.join(save_logs_dir, 'fold'+str(fold), network[i], experiment_code[i]) for i in range(len(experiment_code))]
train_fpaths = [os.path.join(save_logs_folders[i], 'trainlog_gpu0.csv') for i in range(len(save_logs_folders))]
valid_fpaths = [os.path.join(save_logs_folders[i], 'validlog_gpu0.csv') for i in range(len(save_logs_folders))]
legend_lbls = [f'{network[i]}, N = {inputsize[i]}' for i in range(len(network))]
plot_train_logs(train_fpaths, valid_fpaths, legend_lbls)

