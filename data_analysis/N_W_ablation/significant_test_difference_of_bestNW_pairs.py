#%%
'''
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
'''
import numpy as np 
import pandas as pd 
import os 
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy import stats
#%%
# %%

dir = '/data/blobfuse/lymphoma-segmentation-results-new/'
fold = 0
network = 'unet'
inputsize = [96, 128, 160, 192, 224, 256]
W = [128, 160, 192, 192, 224, 256]
experiment_code = [f"{network}_fold{fold}_randcrop{size}_p10_wd1em5_lr2em4" for size in inputsize]
validmetrics_fpaths = [os.path.join(dir, 'valid_metrics', 'fold'+str(fold), network, experiment_code[i], 'validmetrics.csv') for i in range(len(experiment_code))]
validmetrics_dfs = [pd.read_csv(path) for path in validmetrics_fpaths]

p_values_matrix = np.ones((len(validmetrics_dfs),len(validmetrics_dfs)))
sig_values_matrix = np.zeros((len(validmetrics_dfs),len(validmetrics_dfs)))
alpha = 0.05
PVALS = []
for i in range(len(validmetrics_dfs)):
    for j in range(i+1, len(validmetrics_dfs)):
        test_val, p_val = stats.wilcoxon(
            validmetrics_dfs[i]['DSC'], 
            validmetrics_dfs[j]['DSC'], 
            alternative='two-sided'
        )
        p_values_matrix[j][i] = p_val
xtick_labels = [
    r'UNet$(96, 128)$',
    r'UNet$(128, 160)$',
    r'UNet$(160, 192)$',
    r'UNet$(192, 192)$',
    r'UNet$(224, 224)$',
    r'UNet$(256, 256)$',
]
def custom_sci_notation(x, pos):
    return "{:.0e}".format(x).replace("e-0", "e-").replace("e+0", "e")


fig, ax = plt.subplots(1, 2, figsize=(25, 10))
fig.patch.set_facecolor('white')
fig.patch.set_alpha(1)
mask = np.zeros_like(p_values_matrix, dtype=bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(p_values_matrix, 
            mask=mask, linewidths=1, annot=True, square=True, 
            annot_kws={'size': '14'},
            cmap='coolwarm', 
            fmt = '.1e',
            xticklabels=xtick_labels, yticklabels=xtick_labels, 
            ax=ax[0])
cbar = ax[0].collections[0].colorbar
cbar.set_label(f'$p$-values', labelpad=-80, fontsize=20)

ax[0].set_xticklabels(xtick_labels, rotation=90, fontsize=20)
ax[0].set_yticklabels(xtick_labels, rotation=0, fontsize=20)
ax[0].set_title(r'$p$-value heatmap', fontsize=25)

sig_values_matrix = np.where(p_values_matrix <= alpha, 1, 0)

sns.heatmap(sig_values_matrix, 
            mask=mask, linewidths=1, annot=True, square=True, 
            cmap='coolwarm', 
            annot_kws={'size': '15'},
            xticklabels=xtick_labels, yticklabels=xtick_labels, 
            ax=ax[1])
cbar = ax[1].collections[0].colorbar
cbar.set_label('Significance (1: Significant, 0: Not significant)', labelpad=-80, fontsize=20)
ax[1].set_xticklabels(xtick_labels, rotation=90, fontsize=20)
ax[1].set_yticklabels(xtick_labels, rotation=0, fontsize=20)
ax[1].set_title(r'Significance heatmap', fontsize=25)
plt.subplots_adjust(wspace=0.25)
fig.savefig('p_val_significance_plots.png', dpi=500, bbox_inches='tight')
# %%
