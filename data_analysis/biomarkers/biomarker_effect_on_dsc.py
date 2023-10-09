#%%
'''
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
'''
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import os 
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
sys.path.append(PARENT_DIR)
from config import RESULTS_FOLDER
#%%

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
# %%
###################################################################
###################################################################
# plot DSC as a function of % of lower values biomarker removed
def plot_dsc_vs_lower_valued_biomarkers_removed(
        df, 
        biomarker='SUVmean', 
        xaxis = 'values', # percentage
        mean_or_median = 'median', 
        reject_removal_top_percentage=0.10,
        ax = None, 
):
    columns_to_extract = [f'{biomarker}_orig', 'DSC']
    data = df[columns_to_extract].astype(float)
    num_cases = len(data)
    xaxis_values = []
    
    dscs = []
    for i in range(num_cases,int(reject_removal_top_percentage*num_cases), -1):
        df_current = data.nlargest(i, columns=[f"{biomarker}_orig"])
        if xaxis == 'values':
            xaxis_val = df_current[f"{biomarker}_orig"].min()
        elif xaxis == 'percentage': # xaxis == 'percentage
            xaxis_val = 100*(len(data) - len(df_current))/len(data)
        else:
            pass
        xaxis_values.append(xaxis_val)
        if mean_or_median == 'mean':
            dscs.append(np.mean(df_current['DSC']))
        elif mean_or_median == 'median':
            dscs.append(np.median(df_current['DSC']))
        else:
            pass
    ax.plot(xaxis_values, dscs)
    return ax


def plot_dsc_vs_cummulative_threshold(
        df, 
        biomarker='SUVmean', 
        orig_or_pred = 'orig',
        ax = None, 
        stepsize=0.1
):
    columns_to_extract = [f'{biomarker}_{orig_or_pred}', 'DSC']
    data = df[columns_to_extract].astype(float)

    thresholds = []
    median_dscs = []
    min_biomarker = data[f"{biomarker}_{orig_or_pred}"].min()
    quantile85_biomarker = data[f"{biomarker}_{orig_or_pred}"].quantile(0.85)
    
    t = min_biomarker
    
    while t <= quantile85_biomarker:
        thresholds.append(t)
        df_current = data[data[f"{biomarker}_{orig_or_pred}"] >= t]
        median_dscs.append(np.median(df_current['DSC']))
        t += stepsize
    
    ax.plot(thresholds, median_dscs, '-o')
    # ax.set_ylim(0.55, 0.85) # external
    ax.set_ylim(0.6, 0.87) # internal
    min_median_dsc = np.min(median_dscs)
    max_median_dsc = np.max(median_dscs)
    increase = max_median_dsc - min_median_dsc
    return increase

networks = ['UNet', 'SegResNet', 'DynUNet', 'SwinUNETR']
biomarkers = ['SUVmean', 'SUVmax', 'LesionCount', 'TMTV', 'TLG', 'Dmax']
biomarkers_labels = [r'SUV$_{mean}$', r'SUV$_{max}$', 'Number of lesions', 'TMTV', 'TLG', r'D$_{max}$']
stepsizes = [1, 2, 1, 25, 150, 3]
fig, ax = plt.subplots(2, 3, figsize=(19, 16))
fig.patch.set_facecolor('white')
fig.patch.set_alpha(1)
for i in range(len(dfs)):
    for j in range(len(biomarkers)):
        if j == 0 or j == 1 or j == 2:
            increase = plot_dsc_vs_cummulative_threshold(dfs[i], biomarkers[j], 'orig', ax=ax[0][j], stepsize=stepsizes[j])
            # ax[0][j].set_title(f'{biomarkers_labels[j]}', fontsize=30)
            ax[0][j].grid(True, linestyle='dotted')
            ax[0][j].tick_params(axis='both', labelsize=20)
        else:
            increase = plot_dsc_vs_cummulative_threshold(dfs[i], biomarkers[j], 'orig', ax=ax[1][j-3], stepsize=stepsizes[j])
            ax[1][j-3].grid(True, linestyle='dotted')
            ax[1][j-3].tick_params(axis='both', labelsize=20)
        
        if j == 4:
            print(f'{networks[i]}: {biomarkers[j]}: increase = {100*round(increase,2)}')
        
        
        if j == 0:
            ax[0][j].set_xlabel(r'$t_{SUV_{mean}}$', fontsize=30)
        elif j == 1:
            ax[0][j].set_xlabel(r'$t_{SUV_{max}}$', fontsize=30)
        elif j == 2:
            ax[0][j].set_xlabel(r'$t_{L_{g}}$', fontsize=30)
        elif j == 3:
            ax[1][j-3].set_xlabel(r'$t_{TMTV}$' + f' (ml)', fontsize=30)
        elif j == 4:
            ax[1][j-3].set_xlabel(r'$t_{TLG}$' + f' (ml)', fontsize=30)
        elif j == 5:    
            ax[1][j-3].set_xlabel(r'$t_{D_{max}}$' + f' (cm)', fontsize=30)
        
        
        
        # plt.subplots_adjust(top=0.9, wspace=0.2, hspace=0.2)
        
       
    ax[0][0].set_ylabel('DSC', fontsize=30)
    ax[1][0].set_ylabel('DSC', fontsize=30)
ax[0][0].legend(networks, loc='upper left', bbox_to_anchor=(-0.1, -1.4), ncol=4, fontsize=28, markerscale=2)
fig.savefig('dsc_vs_gt_biomarkers_combined.png', dpi=400, bbox_inches='tight')
