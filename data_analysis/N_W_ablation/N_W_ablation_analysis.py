#%%
'''
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
'''
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
# %%
def plot_N_W_ablation(fname, ax=None, clr='red'):
    df = pd.read_csv(fname)
    Ws = list(df['W'].astype(int).values)
    DSCs = list(df['DSC'].astype(float).values)
    max_dsc = np.max(DSCs)
    max_W = Ws[np.argmax(DSCs)]
    ax.plot(Ws, DSCs, '-o', color=clr)
    return ax, max_dsc, max_W

#%%
fnames = [
    'ablationW_N96.csv',
    'ablationW_N128.csv',
    'ablationW_N160.csv',
    'ablationW_N192.csv',
    'ablationW_N224.csv',
    'ablationW_N256.csv'
]


# %%
W_list = [96, 128, 160, 192, 224, 256, 288]
N = [r'$N=96$', r'$N=128$', r'$N=160$', r'$N=192$', r'$N=224$', r'$N=256$']
colors = ['blue', 'orange', 'green', 'red', 'pink', 'brown']
fig, ax = plt.subplots(figsize=(6, 8))
fig.patch.set_facecolor('white')
fig.patch.set_alpha(1)
max_dscs = []
max_Ws = []
for i in range(len(fnames)):
    _, dsc, W = plot_N_W_ablation(fnames[i], ax=ax, clr=colors[i])
    max_dscs.append(dsc)
    max_Ws.append(W)
for i in range(len(max_dscs)):
    ax.scatter(max_Ws[i], max_dscs[i], s=100, facecolor='none', edgecolor=colors[i])
ax.grid(True)
ax.legend(N[0:len(fnames)], fontsize=15)   
ax.set_xlabel(r'Inference window size, $W$', fontsize=20) 
ax.set_ylabel(f'DSC', fontsize=20)
ax.set_xticks(W_list)
ax.set_ylim(0.33, 0.65)
ax.set_title(r'$(N, W)$ ablation', fontsize=25)
fig.savefig('NW_ablation.png', dpi=500, bbox_inches='tight')
plt.show()
# %%
