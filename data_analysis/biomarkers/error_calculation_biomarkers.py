#%%
'''
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
'''
import pandas as pd 
import numpy as np 
import os 
import sys 
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
sys.path.append(PARENT_DIR)
from config import RESULTS_FOLDER
# %%
# internal test set biomarker files
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

inputsize = [N_for_network[network[i]] for i in range(len(network))]
experiment_code = [f"{network[i]}_fold{fold}_randcrop{inputsize[i]}_{extra_features}" for i in range(len(network))]
dir = os.path.join(RESULTS_FOLDER, f'biomarkers')
save_biomakers_dir = [os.path.join(dir, 'fold'+str(fold), network[i], experiment_code[i]) for i in range(len(network))]
fpaths = [os.path.join(d, f'biomarkers.csv') for d in save_biomakers_dir]
dfs_internal = [pd.read_csv(path) for path in fpaths]

dir = os.path.join(RESULTS_FOLDER, f'autopet_biomarkers')
save_biomakers_dir = [os.path.join(dir, 'fold'+str(fold), network[i], experiment_code[i]) for i in range(len(network))]
fpaths = [os.path.join(d, f'biomarkers.csv') for d in save_biomakers_dir]
dfs_external = [pd.read_csv(path) for path in fpaths]

dfs = [pd.concat([dfs_internal[i], dfs_external[i]], axis=0) for i in range(4)]

#%%
def get_gt_pred_biomarkers(df, biomarker='SUVmean'):
    biomarker_values = df[[f'{biomarker}_orig', f'{biomarker}_pred']].astype(float)
    biomarker_values = biomarker_values.dropna()
    return biomarker_values

def get_error(orig, pred, error_type='rmse'):
    if error_type == 'mae':
        diff = pred - orig
        error = np.mean(np.abs(diff))
    elif error_type == 'rmse':
        diff = pred - orig
        error = np.sqrt(np.mean(diff**2))
    elif error_type == 'mape':
        ratio = (pred - orig)/orig
        error = 100*np.mean(np.abs(ratio))
    elif error_type == 'rmspe':
        ratio = (pred - orig)/orig
        error = 100*np.sqrt(np.mean(ratio**2))
    else:
        print('Incorrect error type!')
        return 
    return error
        

def get_errors_bincenters(biomarker_values, biomarker='SUVmean', crossover_point=500.0, error_type = 'rmse'):
    log_min = np.log10(float(biomarker_values[f'{biomarker}_orig'].min()))
    log_max = np.log10(float(crossover_point))
    log_bins = np.logspace(log_min, log_max, 8)
    lin_bins = np.linspace(crossover_point, biomarker_values[f'{biomarker}_orig'].max(), 8)[1:]
    bins_ = np.concatenate((log_bins, lin_bins))
    values, bins_ = np.histogram(biomarker_values[f'{biomarker}_orig'], bins=bins_)
    bins_useful = [bins_[0]]
    for i in range(len(values)):
        if values[i] == 0:
            continue
        else:
            bins_useful.append(bins_[i+1])
    bin_centers, error_values = [], []
    for i in range(len(bins_useful)-2):
        bin_data = biomarker_values[(biomarker_values[f'{biomarker}_orig'] >= bins_useful[i]) & (biomarker_values[f'{biomarker}_orig'] < bins_useful[i+1])]        
        orig, pred = bin_data[f'{biomarker}_orig'], bin_data[f'{biomarker}_pred']
        error = get_error(orig, pred, error_type)
        bin_centers.append((bins_useful[i] + bins_useful[i+1]) / 2)
        error_values.append(error)
    return bin_centers, error_values

def find_index(array, target, tolerance=1e-7):
    for i, value in enumerate(array):
        if abs(value - target) < tolerance:
            return i
    return -1 

def plot_error_vs_gt_biomarker(
    dfs, 
    ax = None, 
    crossover_point = 0,
    biomarker='SUVmean', 
    error_type='rmse', 
    biomarker_label=r'SUV$_{mean}$', 
    biomarker_unit= '', 
    i=0, j=0, rows=0, cols=0
):
    unet_df, segresnet_df, dynunet_df, swinunetr_df = dfs[0], dfs[1], dfs[2], dfs[3]
    unet_biomarkers = get_gt_pred_biomarkers(unet_df, biomarker)
    segresnet_biomarkers = get_gt_pred_biomarkers(segresnet_df, biomarker)
    dynunet_biomarkers = get_gt_pred_biomarkers(dynunet_df, biomarker)
    swinunetr_biomarkers = get_gt_pred_biomarkers(swinunetr_df, biomarker)
    
    unet_bc, unet_error =  get_errors_bincenters(unet_biomarkers, biomarker, crossover_point, error_type)
    segresnet_bc, segresnet_error =  get_errors_bincenters(segresnet_biomarkers, biomarker,crossover_point, error_type)
    dynunet_bc, dynunet_error =  get_errors_bincenters(dynunet_biomarkers, biomarker,crossover_point, error_type)
    swinunetr_bc, swinunetr_error =  get_errors_bincenters(swinunetr_biomarkers, biomarker,crossover_point, error_type)
    
    if biomarker == 'TLG':
        value_cutoff = 2493.8386891606856
        index_cutoff = find_index(unet_bc, value_cutoff)

        unet_mape_avg = unet_error[index_cutoff:]
        segresnet_mape_avg = segresnet_error[index_cutoff:]
        dynunet_mape_avg = dynunet_error[index_cutoff:]
        swinunetr_mape_avg = swinunetr_error[index_cutoff:]
    
        print(unet_mape_avg)
        print(segresnet_mape_avg)
        print(dynunet_mape_avg)
        print(swinunetr_mape_avg)
        print(np.mean([unet_mape_avg, segresnet_mape_avg, dynunet_mape_avg, swinunetr_mape_avg]))
        print('\n')
    ax.plot(unet_bc, unet_error, '-o')
    ax.plot(segresnet_bc, segresnet_error,  '-o')
    ax.plot(dynunet_bc, dynunet_error,  '-o')
    ax.plot(swinunetr_bc, swinunetr_error,  '-o')
    ax.set_yscale('log')
    # ax.set_xscale('log')
    ax.set_xlabel(biomarker_label + ' ' + biomarker_unit ,  fontsize=23)
    ax.tick_params(axis='both', labelsize=15)
    ax.grid(True, linestyle='dotted')
    return ax

#%%
import matplotlib.pyplot as plt 
biomarkers = [['SUVmean', 'SUVmax', 'LesionCount'], ['TMTV', 'TLG', 'Dmax']]
biomarkers_labels = [[r'SUV$_{mean}$', r'SUV$_{max}$', 'Number of lesions'], ['TMTV', 'TLG', r'D$_{max}$']]
biomarkers_units = [['', '', ''], [f' (ml)',  f' (ml)', f' (cm)']]
networks = ['UNet', 'SegResNet', 'DynUNet', 'SwinUNETR']
crossover_points = [[5.0, 20.0, 40],[500, 4000, 20]]
num_bins_list = [[6, 8, 6], [7, 6, 6]]
fig, ax = plt.subplots(2, 3, figsize=(15, 12))
fig.patch.set_facecolor('white')
fig.patch.set_alpha(1)
ERROR_TYPE = 'mape'
for i in range(2):
    for j in range(3):
        plot_error_vs_gt_biomarker(
            dfs, 
            ax = ax[i][j], 
            error_type=ERROR_TYPE, 
            biomarker=biomarkers[i][j], 
            crossover_point = crossover_points[i][j],
            biomarker_label=biomarkers_labels[i][j],
            biomarker_unit = biomarkers_units[i][j],
        )
ax[0][0].legend(networks, fontsize=16)
ax[0][1].set_yscale('linear')
ax[0][0].set_ylabel(f'{ERROR_TYPE.upper()} (%)\n(Internal + External)', fontsize=23)
ax[1][0].set_ylabel(f'{ERROR_TYPE.upper()} (%)\n(Internal + External)', fontsize=23)
fig.savefig(f'{ERROR_TYPE}_vs_gt_biomarkers_combined.png', dpi=500, bbox_inches='tight')
#%%
