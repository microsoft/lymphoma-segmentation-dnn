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
def add_detection_metrics_columns(dfs):
    networks = ['UNet', 'SegResNet', 'DynUNet', 'SwinUNETR']
    for i in range(len(dfs)):
        for j in range(1, 4):
            dfs[i][f'sens_M{j}'] = dfs[i][f'TP_M{j}']/(dfs[i][f'TP_M{j}'] + dfs[i][f'FN_M{j}'])
            dfs[i][f'pres_M{j}'] = dfs[i][f'TP_M{j}']/(dfs[i][f'TP_M{j}'] + dfs[i][f'FP_M{j}'])
            dfs[i][f'missr_M{j}'] = dfs[i][f'FN_M{j}']/(dfs[i][f'TP_M{j}'] + dfs[i][f'FN_M{j}'])
            dfs[i][f'fdr_M{j}'] = dfs[i][f'FP_M{j}']/(dfs[i][f'TP_M{j}'] + dfs[i][f'FP_M{j}'])
    return dfs 

#%%
dfs_internal = [pd.read_csv(path) for path in internal_fpaths]
dfs_internal = shuffle_rows_internal(dfs_internal)

dfs_external = [pd.read_csv(path) for path in external_fpaths]
dfs_internal_copy = dfs_internal.copy()
dfs_external_copy = dfs_external.copy()

dfs_internal1 = add_new_columns_remove_useless_columns(dfs_internal, 'Internal')
dfs_internal2 = add_new_columns(dfs_internal, 'Internal')
dfs_external1 = add_new_columns_remove_useless_columns(dfs_external, 'External')
#%%
dfs_internal1_det = add_detection_metrics_columns(dfs_internal1)
dfs_internal2_det = add_detection_metrics_columns(dfs_internal2)
dfs_external1_det = add_detection_metrics_columns(dfs_external1)
#%%
dfs1 = pd.concat(dfs_internal1_det + dfs_external1_det)
dfs2 = pd.concat(dfs_internal2_det)
# %%

fig, ax = plt.subplots(2, 3, figsize=(22, 14))
fig.patch.set_facecolor('white')
fig.patch.set_alpha(1)
sns.set_style("whitegrid", {'grid.linestyle': '--'})
detection_metric = ['sens', 'FP']
detection_metric_labels = ['Sensitivity', 'False positives']
method = ['M1', 'M2', 'M3']
method_labels = ['Method 1', 'Method 2', 'Method 3']
for i in range(len(detection_metric)):
    for j in range(len(method)):
        plot = sns.boxplot(
            data=dfs1, 
            y=f"{detection_metric[i]}_{method[j]}", x='Network', hue='Test set',
            width=0.5,
            zorder=2,
            hue_order = ['Internal', 'External'],
            palette = sns.color_palette('pastel')[0:2],
            showmeans=True,
            medianprops=dict(color="red", alpha=1),
            meanprops={"marker":"o", 
                       "markerfacecolor":"white", 
                       "markeredgecolor":"black",
                      "markersize":"8"},
            ax=ax[i][j]
        )
        networks = ['UNet', 'SegResNet', 'DynUNet', 'SwinUNETR']
        ax[i][j].set_xticklabels(networks, fontsize=15)
        ax[i][j].tick_params(axis='both', labelsize=15)

        if i == 0:
            ax[i][j].set_title(f'{method_labels[j]}', fontsize=30)
            ax[i][j].margins(x=0.05)
        if j == 0:
            ax[i][j].set_ylabel(f'{detection_metric_labels[i]}', fontsize=25)
        else:
            ax[i][j].set(ylabel=None)
        if i != len(detection_metric) - 1:
            ax[i][j].set(xlabel=None)
        else:
            # ax[i][j].tick_params(axis="x", pad=20)
            ax[i][j].set_xlabel('Network', fontsize=25, labelpad=15)
        if i == 1:
            ax[i][j].set_ylim([-2, 158])

        if i != 1 or j != 0:
            plot.legend_.remove()
        else:
            legend1 = plot.legend(loc='lower left', bbox_to_anchor=(1.29, -0.40), ncol=2)
            legend1 = plot.get_legend()
            for text in legend1.get_texts():
                text.set_fontsize(25)
            legend1.set_title("Test set", prop={'size': 28})  # Adjust the title and size as needed
fig.savefig('detection_boxplots.png', dpi=400, bbox_inches='tight')


# %%

def get_detection_statistics(df, method='M1'):
    sens_mean = df[f'sens_{method}'].mean()
    sens_std = df[f'sens_{method}'].std()
    sens_median = df[f'sens_{method}'].median()
    sens_quantile25 = df[f'sens_{method}'].quantile(q=0.25)
    sens_quantile75 = df[f'sens_{method}'].quantile(q=0.75)

    fp_mean = df[f'FP_{method}'].mean()
    fp_std = df[f'FP_{method}'].std()
    fp_median = df[f'FP_{method}'].median()
    fp_quantile25 = df[f'FP_{method}'].quantile(q=0.25)
    fp_quantile75 = df[f'FP_{method}'].quantile(q=0.75)

    print(f"Sensitivity Mean: {round(sens_mean, 2)} +/- {round(sens_std, 2)}")
    print(f"Sensitivity Median: {round(sens_median, 2)} [{round(sens_quantile25, 2)}, {round(sens_quantile75, 2)}]")
    print(f"FP Mean: {round(fp_mean, 2)} +/- {round(fp_std, 2)}")
    print(f"FP Median: {round(fp_median, 2)} [{round(fp_quantile25, 2)}, {round(fp_quantile75, 2)}]")
    
# %%
networks = ['UNet', 'SegResNet', 'DynUNet', 'SwinUNETR']
method = 'M1'
print('################  Method 1  ##################')
print('############################################')
print('Internal')
print('############################################')
for i in range(len(networks)):
    print(f'##### {networks[i]}: {method} #####')
    get_detection_statistics(dfs_internal1_det[i], method=method)
print('############################################')
print('External')
print('############################################')
for i in range(len(networks)):
    print(f'##### {networks[i]}: {method} #####')
    get_detection_statistics(dfs_external1_det[i], method=method)
print('#############################################')
# %%

method = 'M2'
print('################  Method 2  ##################')
print('############################################')
print('Internal')
print('############################################')
for i in range(len(networks)):
    print(f'##### {networks[i]}: {method} #####')
    get_detection_statistics(dfs_internal1_det[i], method=method)
print('############################################')
print('External')
print('############################################')
for i in range(len(networks)):
    print(f'##### {networks[i]}: {method} #####')
    get_detection_statistics(dfs_external1_det[i], method=method)
print('#############################################')

#%%
method = 'M3'
print('################  Method 3  ##################')
print('############################################')
print('Internal')
print('############################################')
for i in range(len(networks)):
    print(f'##### {networks[i]}: {method} #####')
    get_detection_statistics(dfs_internal1_det[i], method=method)
print('############################################')
print('External')
print('############################################')
for i in range(len(networks)):
    print(f'##### {networks[i]}: {method} #####')
    get_detection_statistics(dfs_external1_det[i], method=method)
print('#############################################')
# %%
