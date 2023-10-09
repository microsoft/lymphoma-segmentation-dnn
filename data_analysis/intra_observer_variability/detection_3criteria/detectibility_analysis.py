#%%
'''
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
'''
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
sys.path.append(PARENT_DIR)
import os
import glob
import pandas as pd
import numpy as np
from metrics.metrics import (
    get_3darray_from_niftipath,
    calculate_patient_level_dice_score, 
    calculate_patient_level_false_positive_volume, 
    calculate_patient_level_false_negative_volume,
    calculate_patient_level_tp_fp_fn,
)
import SimpleITK as sitk
# %%
'''
Easy cases in imagedrive_tools4 (n=35)
Hard cases in imagedrive_tools3 (n=25)
'''
easy_or_hard = 'hard'

if easy_or_hard == 'easy':
    imagedrive_dir = 'imagedrive_tools4'
elif easy_or_hard == 'hard':
    imagedrive_dir = 'imagedrive_tools3'
else:
    print('wrong location\n')
    

gtdir_new = f'/data/blobfuse/{imagedrive_dir}/imagedrive_tools/convert_to_nifti/new_nifti_masks_resampled'
gtpaths_new = sorted(glob.glob(os.path.join(gtdir_new, '*.nii.gz')))
ids = [os.path.basename(path)[:-7] for path in gtpaths_new]
#%%
dir = '/data/blobfuse/lymphoma_lesionsize_split/pmbcl_bccv/all/'
ptdir = os.path.join(dir, 'images')
gtdir = os.path.join(dir, 'labels')
ptpaths = []
gtpaths_old = []
for i in range(len(ids)):
    ptpath = os.path.join(ptdir, f"{ids[i]}_0001.nii.gz")
    gtpath_old = os.path.join(gtdir, f"{ids[i]}.nii.gz")
    ptpaths.append(ptpath)
    gtpaths_old.append(gtpath_old)
# %%
test_imageids = []
test_dscs = []
test_fpvs = []
test_fnvs = []
test_tps_method1 = []
test_fps_method1 = []
test_fns_method1 = []
test_tps_method2 = []
test_fps_method2 = []
test_fns_method2 = []
test_tps_method3 = []
test_fps_method3 = []
test_fns_method3 = []

for i in range(len(gtpaths_old)):
    gtpath = gtpaths_old[i]
    predpath = gtpaths_new[i]
    ptpath = ptpaths[i]
    imageid = os.path.basename(gtpath)[:-7]

    gtarray = get_3darray_from_niftipath(gtpath)
    predarray = get_3darray_from_niftipath(predpath)
    ptarray = get_3darray_from_niftipath(ptpath)

    spacing = sitk.ReadImage(gtpath).GetSpacing() # spacing of gtarray and predarray are the same
    
    dsc = calculate_patient_level_dice_score(gtarray, predarray)
    fpv = calculate_patient_level_false_positive_volume(gtarray, predarray, spacing)
    fnv = calculate_patient_level_false_negative_volume(gtarray, predarray, spacing)
    tp_m1, fp_m1, fn_m1 = calculate_patient_level_tp_fp_fn(gtarray, predarray, method='method1')
    tp_m2, fp_m2, fn_m2 = calculate_patient_level_tp_fp_fn(gtarray, predarray, method='method2', threshold=0.5)
    tp_m3, fp_m3, fn_m3 = calculate_patient_level_tp_fp_fn(gtarray, predarray, method='method3', ptarray=ptarray)

    test_imageids.append(imageid)
    test_dscs.append(dsc)
    test_fpvs.append(fpv)
    test_fnvs.append(fnv)

    test_tps_method1.append(tp_m1)
    test_fps_method1.append(fp_m1)
    test_fns_method1.append(fn_m1)

    test_tps_method2.append(tp_m2)
    test_fps_method2.append(fp_m2)
    test_fns_method2.append(fn_m2)

    test_tps_method3.append(tp_m3)
    test_fps_method3.append(fp_m3)
    test_fns_method3.append(fn_m3)

    print(imageid)
    print(f"DSC: {round(dsc, 4)}")
    print(f"FPV: {round(fpv, 2)} cm^3")
    print(f"FNV: {round(fnv, 2)} cm^3")
    print('\n')

#%%
save_testmetrics_fname = os.path.join(f"{easy_or_hard}_metrics.csv")
data = np.column_stack(
    (
        test_imageids,
        test_dscs, test_fpvs, test_fnvs,
        test_tps_method1, test_fps_method1, test_fns_method1,
        test_tps_method2, test_fps_method2, test_fns_method2,
        test_tps_method3, test_fps_method3, test_fns_method3,
    )
)
col_names = [
    'ImageID',
    'DSC', 'FPV', 'FNV', 
    'TP_M1', 'FP_M1', 'FN_M1', 
    'TP_M2', 'FP_M2', 'FN_M2', 
    'TP_M3', 'FP_M3', 'FN_M3'
]
data_df = pd.DataFrame(data=data, columns=col_names)
data_df.to_csv(save_testmetrics_fname, index=False)
# %%

import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np 
easy_path = 'easy_metrics.csv'
hard_path = 'hard_metrics.csv'

easy_df = pd.read_csv(easy_path)
hard_df = pd.read_csv(hard_path)
#%%
def add_new_columns(df, ImageType = 'Easy'):
    imagetype_col = [ImageType]*len(df)
    new_cols_df = pd.DataFrame(imagetype_col, columns=['ImageType'])
    df = pd.concat([df, new_cols_df], axis=1)
    return df
# %%

easy_df_new = add_new_columns(easy_df, 'Easy')
hard_df_new = add_new_columns(hard_df, 'Hard')

#%%

def add_detection_metrics_columns(df):
    for j in range(1, 4):
        df[f'sens_M{j}'] = df[f'TP_M{j}']/(df[f'TP_M{j}'] + df[f'FN_M{j}'])
        df[f'pres_M{j}'] = df[f'TP_M{j}']/(df[f'TP_M{j}'] + df[f'FP_M{j}'])
        df[f'missr_M{j}'] = df[f'FN_M{j}']/(df[f'TP_M{j}'] + df[f'FN_M{j}'])
        df[f'fdr_M{j}'] = df[f'FP_M{j}']/(df[f'TP_M{j}'] + df[f'FP_M{j}'])
    return df 

#%%
easy_df_det = add_detection_metrics_columns(easy_df_new)
hard_df_det = add_detection_metrics_columns(hard_df_new)

df_det = pd.concat([easy_df_det, hard_df_det], axis=0)
fig, ax = plt.subplots(2, 3, figsize=(14, 10))
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
            data=df_det, 
            y=f"{detection_metric[i]}_{method[j]}", x='ImageType', 
            width=0.2,
            zorder=2,
            palette = sns.color_palette('pastel')[0:2],
            showmeans=True,
            medianprops=dict(color="red", alpha=1),
            meanprops={"marker":"o", 
                        "markerfacecolor":"white", 
                        "markeredgecolor":"black",
                        "markersize":"8"},
            ax=ax[i][j]
        )
       
        ax[i][j].tick_params(axis='both', labelsize=18)
        if i == 0:
            ax[i][j].set_title(f'{method_labels[j]}', fontsize=25)
        if i == 1:
            ax[i][j].set_ylim([-1, 15.5])
        if j == 0:
            ax[i][j].set_ylabel(f'{detection_metric_labels[i]}', fontsize=22)
        else:
            ax[i][j].set(ylabel=None)
        if i == len(detection_metric) - 1:
            ax[i][j].set_xlabel(f'Cases', fontsize=22)
        else:
            ax[i][j].set(xlabel=None)
fig.savefig('intra_observer_variability_detection_boxplots.png', dpi=500, bbox_inches='tight')

#%%
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
method = 'M1'
print('################  Method 1  ##################')
print('############################################')
print('Easy')
get_detection_statistics(easy_df_det, method=method)
print('############################################')
print('Hard')
get_detection_statistics(hard_df_det, method=method)
print('#############################################')

#%%
method = 'M2'
print('################  Method 2 ##################')
print('############################################')
print('Easy')
get_detection_statistics(easy_df_det, method=method)
print('############################################')
print('Hard')
get_detection_statistics(hard_df_det, method=method)
print('#############################################')

#%%
method = 'M3'
print('################  Method 3 ##################')
print('############################################')
print('Easy')
get_detection_statistics(easy_df_det, method=method)
print('############################################')
print('Hard')
get_detection_statistics(hard_df_det, method=method)
print('#############################################')
# %%
