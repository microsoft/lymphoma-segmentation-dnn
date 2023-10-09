#%%
'''
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
'''
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
from scipy.stats import ttest_rel

#%%
easy_path = 'easy_cases_unetdice>0p75/biomarker_analysis_easycases.csv'
hard_path = 'hard_cases_unetdice<0p2/biomarker_analysis_hardcases.csv'
easy_df = pd.read_csv(easy_path)
hard_df = pd.read_csv(hard_path)

dfs = [easy_df, hard_df]
biomarkers = ['SUVmean', 'SUVmax', 'LesionCount', 'TMTV', 'TLG', 'Dmax']
titles = [r'SUV$_{mean}$', r'SUV$_{max}$', f'Number of lesions', f'TMTV (ml)', f'TLG (ml)', r'$D_{max}$ (cm)']
#%%
def plot_biomarker_paired_ttest(data_df, ax = None, biomarker='SUVmean', title=r'SUV$_{mean}$', i=0, j=0, rows=0, cols=0):
    biomarker_old_new = data_df[[f'{biomarker}_old', f'{biomarker}_new']].astype(float)
    biomarker_old_new = biomarker_old_new.dropna()
    old_values = biomarker_old_new.iloc[:,0].values
    new_values = biomarker_old_new.iloc[:,1].values
    
    stat, p_val = ttest_rel(old_values, new_values)
    
    alpha_ = 0.05
    alpha_corrected = alpha_/(6)
    print(alpha_corrected)
    if p_val < alpha_corrected:
        p_value_label = r'$p < \alpha_{corrected}$'
    else:
        p_value_label = r'$p = $' + f"{p_val:.3f}"
   
    if i == 0: 
        clr = 'r'
    else:
        clr = 'b'
    ax.scatter(old_values, new_values, marker='o', s=150, alpha=0.5, color=clr)
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    xlim_diff = ax.get_xlim()[1] - ax.get_xlim()[0]
    ylim_diff = ax.get_ylim()[1] - ax.get_ylim()[0]
    xloc = [6.5, 15, 6.5, 350, 4000, 20]
    yloc1 = [2.7, 5, 2.5, 100, 1200, 8]
    yloc2 = [1.1, 1, 0.5, 0, 0, 2]
    if i == 0:
        ax.text(xloc[j], yloc1[j], "Easy: " + p_value_label, fontsize=18)
    else:
        ax.text(xloc[j], yloc2[j], "Hard: " + p_value_label, fontsize=18)
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=2)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.tick_params(axis='both', labelsize=15)
    ax.grid(True, linestyle='dotted')
    # if i == rows-1: 
    #     ax.set_xlabel(f"Original", fontsize=25)
    # else:
    #     pass
    if j == 0:
        ax.set_ylabel(f"New", fontsize=25)
    else:
        pass
    if i == 0:
        ax.set_title(title, fontsize=30)
    else:
        pass
    # fig.savefig(savepath, dpi=500, bbox_inches='tight')
    return ax

def plot_original_easy_hard_histograms(easy_df, hard_df, ax=None, biomarker='SUVmean', title='', j=0):
    easy_orig = easy_df[f"{biomarker}_old"].astype(float)
    hard_orig = hard_df[f"{biomarker}_old"].astype(float)

    _, bins, _ = ax.hist(easy_orig, bins=20, edgecolor=None, alpha=0.5, color='r')
    _ = ax.hist(hard_orig, bins=bins, edgecolor=None, alpha=0.5, color='b')
    # ax.set_xscale('log')
    ax.set_xlabel('Original', fontsize=25)
    ax.tick_params(axis='both', labelsize=15)
    if j == 0:
        ax.set_ylabel('Frequency', fontsize=25)
        ax.legend(['Easy', 'Hard'], fontsize=18, bbox_to_anchor=[0.25, 1])
    ax.grid(True, linestyle='dotted')
    return ax

#%%
rows, cols = 2, 6
fig, ax = plt.subplots(rows,cols, figsize=(30, 10))
fig.patch.set_facecolor('white')
fig.patch.set_alpha(1)
legend_elements = [[] for i in range(2)]
for i in range(2):
    for j in range(6):
        # plot correlations
        plot_biomarker_paired_ttest(dfs[i], ax=ax[0][j], biomarker=biomarkers[j], title=titles[j], i=i, j=j, rows=rows, cols=cols)
        # legend_elements[i].append(leg_element)
ax[0][0].legend(handles=[plt.scatter([], [], label='Easy', marker='o', s=150, alpha=0.5, color='red'),
                    plt.scatter([], [], label='Hard', marker='o', s=150, alpha=0.5, color='blue')], fontsize=18, labelspacing=0.5, handlelength=0.5, bbox_to_anchor=[0.4, 0.9])

for j in range(6):
    plot_original_easy_hard_histograms(easy_df, hard_df, ax=ax[1][j], biomarker=biomarkers[j], title=titles[j], j=j)

fig.savefig('intra_observer_variability_all_plots.png', dpi=500, bbox_inches='tight')




# %%
