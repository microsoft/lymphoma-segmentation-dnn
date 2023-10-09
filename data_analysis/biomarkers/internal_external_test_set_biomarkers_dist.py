#%%
'''
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
'''
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np 
sns.set_style("whitegrid", {'grid.linestyle': '--'})
# %%
def shuffle_rows_internal(df):
    df_dlbclbccv = df[df['PatientID'].str.startswith('dlbcl-bccv')]
    df_pmbclbccv = df[df['PatientID'].str.startswith('pmbcl-bccv')]
    df_dlbclsmhs = df[df['PatientID'].str.startswith('dlbcl-smhs')]
    df_new = pd.concat([df_dlbclbccv, df_pmbclbccv, df_dlbclsmhs], axis=0)
    df_new.reset_index(drop=True, inplace=True)
    return df_new
# %%
def add_cohort_column(df):
    cohorts = ['dlbcl-bccv', 'pmbcl-bccv', 'dlbcl-smhs']
    cohort_ids = ['DLBCL-BCCV', 'PMBCL-BCCV', 'DLBCL-SMHS']
    cohort_lens = []
    for cohort in cohorts:
        length = len(df[df['PatientID'].str.startswith(cohort)])
        cohort_lens.append(length)
    cohort_column = [[cohort_ids[i]]*cohort_lens[i] for i in range(len(cohort_ids))]
    cohort_column = [item for sublist in cohort_column for item in sublist]
    cohort_df = pd.DataFrame(cohort_column, columns=['CohortID'])
    df_new = pd.concat([df, cohort_df], axis=1)
    df_new.reset_index(drop=True, inplace=True)
    return df_new


def add_new_column(df, txt='Overall', column_name='DSC Group'):
    new_column = [txt]*len(df)
    new_column_df = pd.DataFrame(new_column, columns=[column_name])
    df_new = pd.concat([df, new_column_df], axis=1)
    return df_new

def get_dsclessthan0p2(df):
    df_new = df[df['DSC'] < 0.2]
    df_new.reset_index(drop=True, inplace=True)
    return df_new

def get_dscgreaterthan0p75(df):
    df_new = df[df['DSC'] > 0.75]
    df_new.reset_index(drop=True, inplace=True)
    return df_new

def get_dscbetween0p20p75(df):
    df_new = df[(df['DSC'] >= 0.2) & (df['DSC'] <= 0.75)]
    df_new.reset_index(drop=True, inplace=True)
    return df_new

def plot_gt_biomarkers_distribution(df, biomarker='SUVmean', ax=None):
    
    sns.boxplot(
        data=df, 
        y=f"{biomarker}_orig", x='CohortID', hue='DSC group',
        width=0.5,
        zorder=2,
        palette = [sns.color_palette('pastel')[i] for i in [2, -2, 6, 5]],
        ax=ax,
        showmeans=True,
        medianprops=dict(color="black", alpha=1),
        meanprops={"marker":"o", 
                    "markerfacecolor":"white", 
                    "markeredgecolor":"black",
                    "markersize":"5"},
        
    )
    biomarkers = ['SUVmean', 'SUVmax', 'LesionCount', 'TMTV', 'TLG', 'Dmax']
    biomarker_labels = [r'SUV$_{mean}$', r'SUV$_{max}$', 'Number of lesions', 'TMTV', 'TLG', r'D$_{max}$']
    units = ['', '', '', f'(ml)', r'(ml)', r'(cm)']
    idx = biomarkers.index(biomarker)
    biomarker_label = f"{biomarker_labels[idx]} " + units[idx]
    cohort_labels = ['DLBCL-BCCV', 'PMBCL-BCCV', 'DLBCL-SMHS', 'External']
    # ax.set_title(biomarker_label, fontsize=28)
    ax.set_xlabel('Cohort', fontsize=20)
    ax.set_xticklabels(cohort_labels, fontsize=15)
    ax.tick_params(axis='both', labelsize=15)
    ax.set_xticklabels(cohort_labels, rotation=30, ha='right')
    ax.set_ylabel(biomarker_label, fontsize=20)
    ax.grid(True, linestyle='dotted')
    ax.legend(bbox_to_anchor=(5, -0.3), ncols=4, fontsize=25)
    if biomarker != 'SUVmean':
        ax.legend_.remove()
    else:
        ax.legend(bbox_to_anchor=(3.1, -1.4), ncols=4, fontsize=22)
    return ax


# %%
path = '/data/blobfuse/lymphoma-segmentation-results-new/biomarkers/fold0/unet/unet_fold0_randcrop224_p2_wd1em5_lr2em4/biomarkers.csv'
path_ext = '/data/blobfuse/lymphoma-segmentation-results-new/autopet_biomarkers/fold0/unet/unet_fold0_randcrop224_p2_wd1em5_lr2em4/biomarkers.csv'
df_int = pd.read_csv(path)
df_ext = pd.read_csv(path_ext)
df_int = add_cohort_column(shuffle_rows_internal(df_int))
df_ext = add_new_column(df_ext, 'External', 'CohortID')
df1 = pd.concat([df_int, df_ext], axis=0)
df1.reset_index(drop=True, inplace=True)
df1 = add_new_column(df1, 'Overall', 'DSC group')

#%%
df_int_dsclt0p2 = get_dsclessthan0p2(df_int)
df_ext_dsclt0p2 = get_dsclessthan0p2(df_ext)
df2 = pd.concat([df_int_dsclt0p2, df_ext_dsclt0p2], axis=0)
df2.reset_index(drop=True, inplace=True)
df2 = add_new_column(df2, 'DSC < 0.2', 'DSC group')

#%%
df_int_dscbw0p20p75 = get_dscbetween0p20p75(df_int)
df_ext_dscbw0p20p75 = get_dscbetween0p20p75(df_ext)
df3 = pd.concat([df_int_dscbw0p20p75, df_ext_dscbw0p20p75], axis=0)
df3.reset_index(drop=True, inplace=True)
df3 = add_new_column(df3, r'0.2 $\leq$ DSC $\leq$ 0.75', 'DSC group')

#%%


df_int_dscgt0p75 = get_dscgreaterthan0p75(df_int)
df_ext_dscgt0p75 = get_dscgreaterthan0p75(df_ext)
df4 = pd.concat([df_int_dscgt0p75, df_ext_dscgt0p75], axis=0)
df4.reset_index(drop=True, inplace=True)
df4 = add_new_column(df4, 'DSC > 0.75', 'DSC group')
#%%
df_all = pd.concat([df1, df2, df3, df4], axis=0)
df_all.reset_index(drop=True, inplace=True)
#%%
biomarkers = [['SUVmean', 'SUVmax', 'LesionCount'], ['TMTV', 'TLG', 'Dmax']]
fig, ax = plt.subplots(2, 3, figsize=(20, 12), gridspec_kw = {'wspace':0.25, 'hspace':0.05})
fig.patch.set_facecolor('white')
fig.patch.set_alpha(1)
for i in range(2):
    for j in range(3):
        plot_gt_biomarkers_distribution(df_all, biomarker=biomarkers[i][j], ax=ax[i][j])
        # if i == 0 and j == 2 or i == 1 and j == 0 or i == 1 and j == 1:
        ax[i][j].set_yscale('log')
        if i == 0:
            ax[i][j].set_xlabel('')
            ax[i][j].set_xticks([])
# %%
fig.savefig('internal_external_gt_biomarkers_distribution.png', dpi=400, bbox_inches='tight')
# %%
