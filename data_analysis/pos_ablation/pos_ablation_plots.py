#%%
'''
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
'''
import numpy as np
import pandas as pd 
import os 
import matplotlib.pyplot as plt 
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
sys.path.append(PARENT_DIR)
from config import RESULTS_FOLDER
# %%
dir = '/data/blobfuse/default/lymphoma-segmentation-results-new/logs/fold0/unet'
pos = [1, 2, 4, 6, 8, 10, 12, 14, 16]
experiment_code = [f'unet_fold0_randcrop224_p{p}_wd1em5_lr2em4' for p in pos]
validlogs_paths = [os.path.join(dir, e, 'validdice_gpu0.csv') for e in experiment_code]
dfs = [pd.read_csv(path) for path in validlogs_paths]

max_dscs = []
for df in dfs:
    max_dscs.append(df['Metric'].astype(float).max())
#%%
plt.plot(pos, max_dscs, '-o')

#%%

fold = 0
network = 'unet'
inputsize = 224
pos = [1, 2, 4, 6, 8, 10, 12, 14, 16]
extra_features = [f'p{p}_wd1em5_lr2em4' for p in pos]
experiment_code = [f"{network}_fold{fold}_randcrop{inputsize}_{e}" for e in extra_features]
save_testmetrics_dir = os.path.join(RESULTS_FOLDER, 'valid_metrics')
save_testmetrics_paths = [os.path.join(save_testmetrics_dir, 'fold'+str(fold), network, e, 'validmetrics.csv') for e in experiment_code]


dfs = [pd.read_csv(path) for path in save_testmetrics_paths]

#%%

MEAN_DSCS, MEDIAN_DSCS = [], []
MEAN_FPVS, MEDIAN_FPVS = [], []
MEAN_FNVS, MEDIAN_FNVS = [], []
for df in dfs:
    MEAN_DSCS.append(df['DSC'].astype(float).mean())
    MEAN_FPVS.append(df['FPV'].astype(float).mean())
    MEAN_FNVS.append(df['FNV'].astype(float).mean())
    MEDIAN_DSCS.append(df['DSC'].astype(float).median())
    MEDIAN_FPVS.append(df['FPV'].astype(float).median())
    MEDIAN_FNVS.append(df['FNV'].astype(float).median())


fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].plot(pos, MEAN_DSCS, '-o', color='red')
ax[0].plot(pos, MEDIAN_DSCS, '-o', color='blue')
ax[0].legend(['Mean', 'Median'])
ax[0].set_title('DSC')
ax[0].set_xticks(pos)

ax[1].plot(pos, MEAN_FPVS, '-o', color='red')
ax[1].plot(pos, MEDIAN_FPVS, '-o', color='blue')
ax[1].legend(['Mean', 'Median'])
ax[1].set_title('FPV')
ax[1].set_xticks(pos)

ax[2].plot(pos, MEAN_FNVS, '-o', color='red')
ax[2].plot(pos, MEDIAN_FNVS, '-o', color='blue')
ax[2].legend(['Mean', 'Median'])
ax[2].set_title('FNV')
ax[2].set_xticks(pos)
#%%
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor('white')
fig.patch.set_alpha(1)
ax.plot(pos, MEAN_DSCS, '-o', color='blue')
ax.scatter(pos[np.argmax(MEAN_DSCS)], np.max(MEAN_DSCS), s=150, facecolor='none', edgecolor='blue')
ax.set_xlabel(r'$pos$', fontsize=20)
ax.set_ylabel('DSC', fontsize=20)
ax.set_xticks(pos)
ax.grid(True)
fig.savefig('pos_ablation', dpi=500, bbox_inches='tight')
plt.show()


# %%
