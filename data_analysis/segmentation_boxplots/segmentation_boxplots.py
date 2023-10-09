#%%
'''
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
'''
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

# %%
internal_fpaths = [
    '/data/blobfuse/lymphoma-segmentation-results-new/test_metrics/fold0/unet/unet_fold0_randcrop224_p2_wd1em5_lr2em4/testmetrics.csv',
    '/data/blobfuse/lymphoma-segmentation-results-new/test_metrics/fold0/segresnet/segresnet_fold0_randcrop192_p2_wd1em5_lr2em4/testmetrics.csv',
    '/data/blobfuse/lymphoma-segmentation-results-new/test_metrics/fold0/dynunet/dynunet_fold0_randcrop160_p2_wd1em5_lr2em4/testmetrics.csv',
    '/data/blobfuse/lymphoma-segmentation-results-new/test_metrics/fold0/swinunetr/swinunetr_fold0_randcrop128_p2_wd1em5_lr2em4/testmetrics.csv',
]
external_fpaths = [
    '/data/blobfuse/lymphoma-segmentation-results-new/autopet_test_metrics/fold0/unet/unet_fold0_randcrop224_p2_wd1em5_lr2em4/testmetrics.csv',
    '/data/blobfuse/lymphoma-segmentation-results-new/autopet_test_metrics/fold0/segresnet/segresnet_fold0_randcrop192_p2_wd1em5_lr2em4/testmetrics.csv',
    '/data/blobfuse/lymphoma-segmentation-results-new/autopet_test_metrics/fold0/dynunet/dynunet_fold0_randcrop160_p2_wd1em5_lr2em4/testmetrics.csv',
    '/data/blobfuse/lymphoma-segmentation-results-new/autopet_test_metrics/fold0/swinunetr/swinunetr_fold0_randcrop128_p2_wd1em5_lr2em4/testmetrics.csv',
]

#%%
def shuffle_rows_internal(dfs):
    for i in range(len(dfs)):
        df_dlbclbccv = dfs[i][dfs[i]['CohortID'] == 'dlbcl-bccv']
        df_pmbclbccv = dfs[i][dfs[i]['CohortID'] == 'pmbcl-bccv']
        df_dlbclsmhs = dfs[i][dfs[i]['CohortID'] == 'dlbcl-smhs']
        dfs[i] = pd.concat([df_dlbclbccv, df_pmbclbccv, df_dlbclsmhs], axis=0)
        dfs[i] = dfs[i].reset_index(drop=True)
    return dfs
#%%
def add_new_columns_remove_useless_columns(dfs, TestSet='Internal'):
    networks = ['UNet', 'SegResNet', 'DynUNet', 'SwinUNETR']
    useful_cols = [
    'ImageID',
    'DSC', 'FPV', 'FNV', 
    'TP_M1', 'FP_M1', 'FN_M1', 
    'TP_M2', 'FP_M2', 'FN_M2', 
    'TP_M3', 'FP_M3', 'FN_M3']
    dfs_useful = []

    for i in range(len(dfs)):
        testset_col = np.array([TestSet]*len(dfs[i]))
        network_col = np.array([networks[i]]*len(dfs[i]))
        new_cols = np.column_stack((testset_col, network_col))
        new_cols_df = pd.DataFrame(new_cols, columns=['Test set', 'Network'])
        df_useful = dfs[i][useful_cols]
        dfs_useful.append(pd.concat([df_useful, new_cols_df], axis=1))
    return dfs_useful

def add_new_columns(dfs, TestSet='Internal'):
    networks = ['UNet', 'SegResNet', 'DynUNet', 'SwinUNETR']
    dfs_useful = []
    for i in range(len(dfs)):
        testset_col = np.array([TestSet]*len(dfs[i]))
        network_col = np.array([networks[i]]*len(dfs[i]))
        new_cols = np.column_stack((testset_col, network_col))
        new_cols_df = pd.DataFrame(new_cols, columns=['Test set', 'Network'])
        df_useful = dfs[i]
        dfs_useful.append(pd.concat([df_useful, new_cols_df], axis=1))
    return dfs_useful
# %%
dfs_internal = [pd.read_csv(path) for path in internal_fpaths]
dfs_internal = shuffle_rows_internal(dfs_internal)

dfs_external = [pd.read_csv(path) for path in external_fpaths]
dfs_internal_copy = dfs_internal.copy()
dfs_external_copy = dfs_external.copy()
dfs_internal1 = add_new_columns_remove_useless_columns(dfs_internal, 'Internal')
dfs_internal2 = add_new_columns(dfs_internal, 'Internal')
dfs_external1 = add_new_columns_remove_useless_columns(dfs_external, 'External')
#%%
dfs1 = pd.concat(dfs_internal1 + dfs_external1)
dfs2 = pd.concat(dfs_internal2)
# %%
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor('white')
fig.patch.set_alpha(1)
sns.set_style("whitegrid", {'grid.linestyle': '--'})
plot1 = sns.boxplot(
    data=dfs1, 
    y='DSC', x='Network', hue='Test set',
    width=0.5,
    zorder=2,
    hue_order = ['Internal', 'External'],
    palette = sns.color_palette('pastel')[0:2],
    ax=ax[0],
    showmeans=True,
    medianprops=dict(color="red", alpha=1),
    meanprops={"marker":"o", 
                "markerfacecolor":"white", 
                "markeredgecolor":"black",
                "markersize":"8"},
)
legend1 = plot1.legend(loc='lower left', bbox_to_anchor=(0.20, -0.35), ncol=2)
legend1 = plot1.get_legend()
for text in legend1.get_texts():
    text.set_fontsize(14)
legend1.set_title("Test set", prop={'size': 15})  # Adjust the title and size as needed
ax[0].set_xlabel('Network', fontsize=20)
ax[0].set_ylabel('DSC', fontsize=20)
networks = ['UNet', 'SegResNet', 'DynUNet', 'SwinUNETR']
ax[0].set_xticklabels(networks, fontsize=14)

plot2 = sns.boxplot(
    data=dfs2, 
    y='DSC', x='Network', hue='CohortID',
    width=0.6,
    zorder=2,
    hue_order = ['dlbcl-bccv', 'pmbcl-bccv', 'dlbcl-smhs'],
    palette = [sns.color_palette('pastel')[i] for i in [2, -2, 6]],
    showmeans=True,
    medianprops=dict(color="red", alpha=1),
    meanprops={"marker":"o", 
                "markerfacecolor":"white", 
                "markeredgecolor":"black",
                "markersize":"8"},
    ax=ax[1]
)
handles2, labels2 = plot2.get_legend_handles_labels()
new_labels = ['DLBCL-BCCV', 'PMBCL-BCCV', 'DLBCL-SMHS']
label_mapping = {old_label: new_label for old_label, new_label in zip(labels2, new_labels)}
for i, label in enumerate(labels2):
    labels2[i] = label_mapping.get(label, label)

legend2 = plot2.legend(handles2, labels2, loc='lower left', bbox_to_anchor=(-0.135, -0.35), ncol=3)
legend2 = plot2.get_legend()
for text in legend2.get_texts():
    text.set_fontsize(14)
legend2.set_title("Cohorts (Internal test set)", prop={'size': 15})  # Adjust the title and size as needed
ax[1].set_xlabel('Network', fontsize=20)
networks = ['UNet', 'SegResNet', 'DynUNet', 'SwinUNETR']
ax[1].set_xticklabels(networks, fontsize=14)
ax[1].set_ylabel('')

fig.savefig('dsc_dist_on_test_sets.png', dpi=500, bbox_inches='tight')# %%

#%%

dfs3 = pd.concat(dfs_internal2)
fig, ax = plt.subplots(figsize=(45, 8))
sns.boxplot(
    data=dfs3,
    width=0.6, dodge=False,
    saturation=0.5,
    linewidth=1,
    y='DSC', x='ImageID', hue='CohortID',
    palette = [sns.color_palette('pastel')[i] for i in [2, -2, 6]],
    ax=ax
)
plot3 = sns.stripplot(
    data=dfs3,
    dodge=False, jitter=False,
    size=10,
    y='DSC', x='ImageID', hue='Network',
    # palette = {'UNet': 'black', 'SegResNet': 'red', 'DynUNet': 'green', 'SwinUNETR': 'aqua'},
    ax=ax
)
imageid_labels = np.arange(1, 89)
ax.set_xticklabels(imageid_labels, fontsize=20, rotation=90)
handles3, labels3 = plot3.get_legend_handles_labels()
new_labels = ['DLBCL-BCCV', 'PMBCL-BCCV', 'DLBCL-SMHS', 'UNet', 'SegResNet', 'DynUNet', 'SwinUNETR']
label_mapping = {old_label: new_label for old_label, new_label in zip(labels3, new_labels)}
for i, label in enumerate(labels3):
    labels3[i] = label_mapping.get(label, label)

legend3 = plot3.legend(handles3, labels3, loc='lower left', bbox_to_anchor=(0.259, -0.32), ncol=7, markerscale=2)
legend3 = plot3.get_legend()
for text in legend3.get_texts():
    text.set_fontsize(22)
legend3.set_title("Cohorts and networks", prop={'size': 23}) 
ax.set_xlabel('Internal test cases', fontsize=25)
ax.set_ylabel('DSC', fontsize=25)
ax.vlines(x=18.5, ymin=-1, ymax=1, linestyle='dotted')
ax.vlines(x=43.5, ymin=-1, ymax=1, linestyle='dotted')

plt.yticks(fontsize=20)
plt.ylim([-0.05, 1])
fig.savefig('dsc_dist_on_internal_test_set_with_respect_to_network.png', dpi=500, bbox_inches='tight')# %%
# %%


