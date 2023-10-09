#%%
'''
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
'''
import numpy as np 
import pandas as pd 
import glob 
import os 
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
sys.path.append(PARENT_DIR)
from config import RESULTS_FOLDER
import matplotlib.pyplot as plt 
import seaborn as sns 
from scipy.stats import ttest_rel, ttest_ind, wilcoxon, mannwhitneyu, shapiro
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

#%%
#%%
def check_normality(dfs):
    networks = ['UNet', 'SegResNet', 'DynUNet', 'SwinUNETR']
    biomarkers = ['SUVmean', 'SUVmax', 'LesionCount', 'TMTV', 'TLG', 'Dmax']
    fig, ax = plt.subplots(4, 6, figsize=(30, 20))
    normal_notnormal_matrix = np.zeros((2, len(dfs), len(biomarkers)))
    p_val_matrix = np.zeros((2, len(dfs), len(biomarkers)))
    shapiro_stat_matrix = np.zeros((2, len(dfs), len(biomarkers)))
    for i in range(len(dfs)):
        for j in range(len(biomarkers)):
            orig = list(dfs[i][f"{biomarkers[j]}_orig"].values)
            pred = list(dfs[i][f"{biomarkers[j]}_pred"].values)
            stat0, p_val0 = shapiro(orig)
            stat1, p_val1 = shapiro(pred)
            p_val_matrix[0][i][j] = p_val0
            p_val_matrix[1][i][j] = p_val1
            shapiro_stat_matrix[0][i][j] = stat0
            shapiro_stat_matrix[1][i][j] = stat1

            if p_val0 < 0.05:
                normal_notnormal_matrix[0][i][j] = False
            else:
                normal_notnormal_matrix[0][i][j] = True

            if p_val1 < 0.05:
                normal_notnormal_matrix[1][i][j] = False
            else:
                normal_notnormal_matrix[1][i][j] = True
            ax[i][j].hist(orig, color='red', alpha=0.5)
            ax[i][j].hist(pred, color='blue', alpha=0.5)
            ax[i][j].legend(['GT', 'Pred'], fontsize=15)
            if i == 0:
                ax[i][j].set_title(biomarkers[j], fontsize=25)
            if j == 0:
                ax[i][j].set_ylabel(networks[i], fontsize=25)
    return normal_notnormal_matrix, p_val_matrix, shapiro_stat_matrix

#%%
normal_notnormal_matrix, p_val_matrix, shapiro_stat_matrix = check_normality(dfs)

#%%
def check_difference_distribution(dfs):
    networks = ['UNet', 'SegResNet', 'DynUNet', 'SwinUNETR']
    biomarkers = ['SUVmean', 'SUVmax', 'LesionCount', 'TMTV', 'TLG', 'Dmax']
    fig, ax = plt.subplots(4, 6, figsize=(30, 20))
   
    median_matrix = np.zeros((len(dfs), len(biomarkers)))
    for i in range(len(dfs)):
        for j in range(len(biomarkers)):
            orig = np.array(dfs[i][f"{biomarkers[j]}_orig"].values)
            pred = np.array(dfs[i][f"{biomarkers[j]}_pred"].values)
            diff = pred - orig
            median = round(np.median(diff),2)
            median_matrix[i][j] = median
            ax[i][j].hist(diff, bins=30, color='red', alpha=0.5)
            ax[i][j].legend([f'Diff'], fontsize=15)
            if i == 0:
                ax[i][j].set_title(biomarkers[j], fontsize=25)
            if j == 0:
                ax[i][j].set_ylabel(networks[i], fontsize=25)
    return median_matrix

median_matrix = check_difference_distribution(dfs)

#%%
def check_wilcoxon_signed_rank_test(dfs):
    networks = ['UNet', 'SegResNet', 'DynUNet', 'SwinUNETR']
    biomarkers = ['SUVmean', 'SUVmax', 'LesionCount', 'TMTV', 'TLG', 'Dmax']
    fig, ax = plt.subplots(4, 6, figsize=(30, 20))
    different_same_matrix = np.zeros((len(dfs), len(biomarkers)))
    p_val_matrix = np.zeros((len(dfs), len(biomarkers)))
    stat_matrix = np.zeros((len(dfs), len(biomarkers)))

    for i in range(len(dfs)):
        for j in range(len(biomarkers)):
            orig = list(dfs[i][f'{biomarkers[j]}_orig'].values)
            pred = list(dfs[i][f'{biomarkers[j]}_pred'].values)
            stat, p_val = wilcoxon(orig, pred)
            stat_matrix[i][j] = stat
            p_val_matrix[i][j] = p_val
            if p_val < 0.05:
                different_same_matrix[i][j] = 0
            else:
                different_same_matrix[i][j] = 1
            ax[i][j].scatter(orig, pred, alpha=0.5, s=80)
            lims = [
            np.min([ax[i][j].get_xlim(), ax[i][j].get_ylim()]),  # min of both axes
            np.max([ax[i][j].get_xlim(), ax[i][j].get_ylim()]),  # max of both axes
            ]
            
            ax[i][j].plot(lims, lims, 'k-', alpha=0.75, zorder=2)
            ax[i][j].set_aspect('equal')
            ax[i][j].set_xlim(lims)
            ax[i][j].set_ylim(lims)
            if i == 0:
                ax[i][j].set_title(biomarkers[j], fontsize=25)
            if j == 0:
                ax[i][j].set_ylabel(f"{networks[i]}\nPred", fontsize=25)
            if i == len(dfs)-1:
                ax[i][j].set_xlabel(f"GT", fontsize=25)
            stat_label = f"r = " + f"{stat:.2f}"
            if p_val < 0.05:
                p_val_label = 'p-value < 0.05'
            else:
                p_val_label = 'p-value = ' + f"{p_val:.1e}"
            legend_elements = [stat_label + '\n' + p_val_label]
            ax[i][j].legend(legend_elements, fontsize=18, handlelength=0, markerscale=0, loc='lower right')
    return different_same_matrix, stat_matrix, p_val_matrix
#%%
different_same_matrix, stat_matrix, p_val_matrix = check_wilcoxon_signed_rank_test(dfs)

def check_paired_ttest_networks_segregated(dfs, ttest_type='relative'):
    networks = ['UNet', 'SegResNet', 'DynUNet', 'SwinUNETR']
    biomarkers = ['SUVmean', 'SUVmax', 'LesionCount', 'TMTV', 'TLG', 'Dmax']
    fig, ax = plt.subplots(4, 6, figsize=(30, 20))
    different_same_matrix = np.zeros((len(dfs), len(biomarkers)))
    p_val_matrix = np.zeros((len(dfs), len(biomarkers)))
    stat_matrix = np.zeros((len(dfs), len(biomarkers)))
    alpha_ = 0.05
    alpha_corrected = alpha_/(len(networks)*len(biomarkers))
    for i in range(len(dfs)):
        for j in range(len(biomarkers)):
            orig = list(dfs[i][f'{biomarkers[j]}_orig'].values)
            pred = list(dfs[i][f'{biomarkers[j]}_pred'].values)
            if ttest_type == 'relative':
                stat, p_val = ttest_rel(orig, pred)
            elif ttest_type == 'independent':
                stat, p_val = ttest_ind(orig, pred)
            stat_matrix[i][j] = stat
            p_val_matrix[i][j] = p_val
            if p_val < alpha_corrected:
                different_same_matrix[i][j] = 0
            else:
                different_same_matrix[i][j] = 1
            ax[i][j].scatter(orig, pred, alpha=0.5, s=100)
            lims = [
            np.min([ax[i][j].get_xlim(), ax[i][j].get_ylim()]),  # min of both axes
            np.max([ax[i][j].get_xlim(), ax[i][j].get_ylim()]),  # max of both axes
            ]
            
            ax[i][j].plot(lims, lims, 'k-', alpha=0.75, zorder=0)
            ax[i][j].set_aspect('equal')
            ax[i][j].set_xlim(lims)
            ax[i][j].set_ylim(lims)
            if i == 0:
                ax[i][j].set_title(biomarkers[j], fontsize=25)
            if j == 0:
                ax[i][j].set_ylabel(f"{networks[i]}\nPredicted", fontsize=25)
            if i == len(dfs)-1:
                if j == 3 or j == 4:
                    ax[i][j].set_xlabel(r"Ground truth (cm$^3$)", fontsize=25)
                elif j == 5:
                    ax[i][j].set_xlabel(f"Ground truth (cm)", fontsize=25)
                else:
                    ax[i][j].set_xlabel(f"Ground truth", fontsize=25)
            stat_label = f"r = " + f"{stat:.2f}"
            if p_val < alpha_corrected:
                p_val_label = f'p-value < {alpha_corrected:.2e}'
            else:
                p_val_label = 'p-value = ' + f"{p_val:.2e}"
            legend_elements = [stat_label + '\n' + p_val_label]
            ax[i][j].legend(legend_elements, fontsize=18, handlelength=0, markerscale=0, loc='lower right')
    return different_same_matrix, stat_matrix, p_val_matrix
#%%
different_same_matrix1, stat_matrix1, p_val_matrix1 = check_paired_ttest_networks_segregated(dfs, ttest_type = 'relative')

#%%
def check_paired_ttest_networks_aggregated(dfs, ttest_type='relative'):
    networks = ['UNet', 'SegResNet', 'DynUNet', 'SwinUNETR']
    biomarkers = ['SUVmean', 'SUVmax', 'LesionCount', 'TMTV', 'TLG', 'Dmax']
    biomarkers_labels = [r'SUV$_{mean}$', r'SUV$_{max}$', 'Number of lesions', 'TMTV', 'TLG', r'D$_{max}$']
    legend_locs = ['upper left', 'lower right', 'upper left', 'lower right', 'lower right', 'lower right']
    biomarkers_units = ['', '', '', f' (ml)',  f' (ml)', f' (cm)']
    fig, ax = plt.subplots(2, 3, figsize=(24, 15))
    fig.patch.set_facecolor('white')
    fig.patch.set_alpha(1)
    different_same_matrix = np.zeros((len(dfs), len(biomarkers)))
    p_val_matrix = np.zeros((len(dfs), len(biomarkers)))
    stat_matrix = np.zeros((len(dfs), len(biomarkers)))
    alpha_ = 0.05
    alpha_corrected = alpha_/(len(networks)*len(biomarkers))
    for i in range(len(dfs)):
        for j in range(len(biomarkers)):
            orig = list(dfs[i][f'{biomarkers[j]}_orig'].values)
            pred = list(dfs[i][f'{biomarkers[j]}_pred'].values)
            if ttest_type == 'relative':
                stat, p_val = ttest_rel(orig, pred, nan_policy='omit')
            elif ttest_type == 'independent':
                stat, p_val = ttest_ind(orig, pred, nan_policy='omit')
            stat_matrix[i][j] = stat
            p_val_matrix[i][j] = p_val
            if p_val < alpha_corrected:
                different_same_matrix[i][j] = 0
            else:
                different_same_matrix[i][j] = 1
            if j == 0 or j == 1 or j == 2:
                ax[0][j].scatter(orig, pred, alpha=0.5, s=100)
            else:
                ax[1][j-3].scatter(orig, pred, alpha=0.5, s=100)
            
            # if i == 0:
            #     ax[j].set_title(biomarkers_labels[j], fontsize=30)
            if j == 0 or j == 1 or j == 2:
                ax[0][j].set_ylabel(biomarkers_labels[j] + biomarkers_units[j] + ' (predicted)',  fontsize=23)
                ax[0][j].set_xlabel(biomarkers_labels[j] + biomarkers_units[j] + ' (ground truth)',  fontsize=23)
            else:
                ax[1][j-3].set_ylabel(biomarkers_labels[j] + biomarkers_units[j] + ' (predicted)', fontsize=23)
                ax[1][j-3].set_xlabel(biomarkers_labels[j] + biomarkers_units[j] + ' (ground truth)', fontsize=23)
           
    for j in range(len(biomarkers)):
        p_val_n1, p_val_n2, p_val_n3, p_val_n4 = p_val_matrix[:, j]
        print(p_val_n1)
        
        if p_val_n1 < alpha_corrected:
            p_val_n1_label = r'UNet: $p < \alpha_{corrected}$'
        else:
            p_val_n1_label = r'UNet: $p = $' + f"{p_val_n1:.1e}" 
        if p_val_n2 < alpha_corrected:
            p_val_n2_label = r'SegResNet: $p < \alpha_{corrected}$'
        else:
            p_val_n2_label = r'SegResNet: $p = $' + f"{p_val_n2:.1e}" 
        if p_val_n3 < alpha_corrected:
            p_val_n3_label = r'DynUNet: $p < \alpha_{corrected}$'
        else:
            p_val_n3_label = r'DynUNet: $p = $' + f"{p_val_n3:.1e}" 
        if p_val_n4 < alpha_corrected:
            p_val_n4_label = r'SwinUNETR: $p < \alpha_{corrected}$'
        else:
            p_val_n4_label = r'SwinUNETR: $p = $' + f"{p_val_n4:.1e}" 
        legend_elements = [p_val_n1_label, p_val_n2_label, p_val_n3_label, p_val_n4_label]
        if j == 0 or j == 1 or j == 2:
            ax[0][j].legend(legend_elements, fontsize=15, loc=legend_locs[j])
            lims = [
                np.min([ax[0][j].get_xlim(), ax[0][j].get_ylim()]),  # min of both axes
                np.max([ax[0][j].get_xlim(), ax[0][j].get_ylim()]),  # max of both axes
                ]    
            ax[0][j].plot(lims, lims, 'k-', alpha=0.75, zorder=0)
            ax[0][j].set_aspect('equal')
            ax[0][j].set_xlim(lims)
            ax[0][j].set_ylim(lims)
            ax[0][j].tick_params(axis='both', labelsize=15)
            ax[0][j].grid(True, linestyle='dotted')
        else:
            ax[1][j-3].legend(legend_elements, fontsize=14, loc=legend_locs[j])
            lims = [
                np.min([ax[1][j-3].get_xlim(), ax[1][j-3].get_ylim()]),  # min of both axes
                np.max([ax[1][j-3].get_xlim(), ax[1][j-3].get_ylim()]),  # max of both axes
                ]    
            ax[1][j-3].plot(lims, lims, 'k-', alpha=0.75, zorder=0)
            ax[1][j-3].set_aspect('equal')
            ax[1][j-3].set_xlim(lims)
            ax[1][j-3].set_ylim(lims)
            ax[1][j-3].tick_params(axis='both', labelsize=15)
            ax[1][j-3].grid(True, linestyle='dotted')
    fig.savefig('reproducibility_of_biomarkers_external.png', dpi=400, bbox_inches='tight')
    return different_same_matrix, stat_matrix, p_val_matrix
#%%
different_same_matrix1, stat_matrix1, p_val_matrix1 = check_paired_ttest_networks_aggregated(dfs_internal, ttest_type = 'relative')
#%%
different_same_matrix1, stat_matrix1, p_val_matrix1 = check_paired_ttest_networks_aggregated(dfs_external, ttest_type = 'relative')
#%%
def plot_biomarker_ttest_rel(data_df, biomarker='SUVmean', title=r'SUV$_{mean}$', ax=None, unit='', savepath=''):
    biomarker_orig_pred = data_df[[f'{biomarker}_orig', f'{biomarker}_pred']].astype(float)
    biomarker_orig_pred = biomarker_orig_pred.dropna()
    orig_values = biomarker_orig_pred.iloc[:,0].values
    pred_values = biomarker_orig_pred.iloc[:,1].values
    # Calculate correlation coefficient and p-value
    t_statistic, p_value = mannwhitneyu(orig_values, pred_values, method='exact')
    print(f"r = {t_statistic}, p={p_value}")
    t_statistic_label = r"$r = $" + f"{t_statistic:.2f}"
    if p_value < 0.05:
        p_value_label = r'p-value < 0.05'
    else:
        p_value_label = r'p-value = ' + f"{p_value:.1e}"

    legend_elements = [t_statistic_label + '\n' + p_value_label] #+ '\n' + diff_of_means_label]
    
    ax.scatter(orig_values, pred_values, marker='o', s=70, alpha=0.5)
    return ax, p_value
    
#%%
networks = ['UNet', 'SegResNet', 'DynUNet', 'SwinUNETR']
biomarkers = ['SUVmean', 'SUVmax', 'LesionCount', 'TMTV', 'TLG', 'Dmax']
biomarker_labels = [r'SUV$_{mean}$', r'SUV$_{max}$', 'Number of lesions', 'TMTV', 'TLG', r'D$_{max}$']
xlabels = ['Ground truth', 'Ground truth', 'Ground truth', r'Ground truth (cm$^3$)', r'Ground truth (cm$^3$)', r'Ground truth (cm)']
fig, ax = plt.subplots(1, 6, figsize=(30,20))
fig.patch.set_facecolor('white')
fig.patch.set_alpha(1)

p_val_matrix = np.zeros((len(networks), len(biomarkers)))
for i in range(len(networks)):
    for j in range(len(biomarkers)):
        ax[j], p_val_matrix[i][j] = plot_biomarker_ttest_rel(dfs[i], biomarkers[j], biomarker_labels[j], ax=ax[j], unit='')
        # if i == 0:
        ax[j].set_title(biomarker_labels[j], fontsize=24)
        if j == 0:
            ax[j].set_ylabel(f"Predicted", fontsize=21)
        # if i == len(networks)-1:
        ax[j].set_xlabel(xlabels[j], fontsize=21)

ax[0].legend(networks, loc='upper left', bbox_to_anchor=(0.3, -0.5), ncol=4, fontsize=20)
for j in range(len(biomarkers)):
    lims = [
        np.min([ax[j].get_xlim(), ax[j].get_ylim()]),  # min of both axes
        np.max([ax[j].get_xlim(), ax[j].get_ylim()]),  # max of both axes
    ]

    ax[j].plot(lims, lims, 'k-', alpha=0.75, zorder=2)
    ax[j].set_aspect('equal')
    ax[j].set_xlim(lims)
    ax[j].set_ylim(lims)
# %%
from scipy.stats import shapiro
networks = ['UNet', 'SegResNet', 'DynUNet', 'SwinUNETR']
biomarkers = ['SUVmean', 'SUVmax', 'LesionCount', 'TMTV', 'TLG', 'Dmax']
orig_pred = ['orig', 'pred']
normal_notnormal = []
for i in range(len(networks)):
    for j in range(len(biomarkers)):
        for k in range(len(orig_pred)):
            print(f'Network: {networks[i]}')
            print(f'Biomarker {biomarkers[j]} - {orig_pred[k]}')
            data = dfs[i][f'{biomarkers[j]}_{orig_pred[k]}']
            stat, p_val = shapiro(data)
            if p_val > 0.05:
                print(f'Normal')
                normal_notnormal.append('normal')
            else:
                print(f"Not normal")
                normal_notnormal.append('notnormal')
            print('\n')
# %%
def plot_differences(data_df, biomarker='SUVmean', title=r'SUV$_{mean}$', ax=None, unit='', savepath=''):
    biomarker_orig_pred = data_df[[f'{biomarker}_orig', f'{biomarker}_pred']].astype(float)
    biomarker_orig_pred = biomarker_orig_pred.dropna()
    orig_values = biomarker_orig_pred.iloc[:,0].values
    pred_values = biomarker_orig_pred.iloc[:,1].values
    diff_values = pred_values - orig_values
    ax.hist(diff_values)
    return ax

fig, ax = plt.subplots(4, 6, figsize=(30,20))
fig.patch.set_facecolor('white')
fig.patch.set_alpha(1)
for i in range(len(networks)):
    for j in range(len(biomarkers)):
        plot_differences(dfs[i], biomarkers[j], biomarker_labels[j], ax=ax[i][j], unit='')
        if i == 0:
            ax[i][j].set_title(biomarker_labels[j], fontsize=24)
        if j == 0:
            ax[i][j].set_ylabel(f"{networks[i]} prediction", fontsize=21)
        if i == len(networks)-1:
            ax[i][j].set_xlabel(xlabels[j], fontsize=21)
# %%
