#%%
'''
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
'''
import numpy as np 
import pandas as pd 
import SimpleITK as sitk 
import os 
from glob import glob 
import sys 
import argparse
config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(config_dir)
from config import RESULTS_FOLDER
from metrics.metrics import (
    get_3darray_from_niftipath,
    calculate_patient_level_dice_score,
    calculate_patient_level_false_positive_volume,
    calculate_patient_level_false_negative_volume,
    calculate_patient_level_tp_fp_fn
)

def get_spacing_from_niftipath(path):
    image = sitk.ReadImage(path)
    return image.GetSpacing()

def get_column_statistics(col):
    mean = col.mean()
    std = col.std()
    median = col.median()
    quantile25 = col.quantile(q=0.25)
    quantile75 = col.quantile(q=0.75)
    return (mean, std, median, quantile25, quantile75)

def get_prediction_statistics(data_df):
    dsc_stats = get_column_statistics(data_df['DSC'].astype(float))
    fpv_stats = get_column_statistics(data_df['FPV'].astype(float))
    fnv_stats = get_column_statistics(data_df['FNV'].astype(float))
    
    c1_sensitivity = data_df[f'TP_C1']/(data_df[f'TP_C1'] + data_df[f'FN_C1'])
    c2_sensitivity = data_df[f'TP_C2']/(data_df[f'TP_C2'] + data_df[f'FN_C2'])
    c3_sensitivity = data_df[f'TP_C3']/(data_df[f'TP_C3'] + data_df[f'FN_C3'])
    sens_c1_stats = get_column_statistics(c1_sensitivity)
    sens_c2_stats = get_column_statistics(c2_sensitivity)
    sens_c3_stats = get_column_statistics(c3_sensitivity)
    
    fp_c1_stats = get_column_statistics(data_df['FP_M1'].astype(float))
    fp_c2_stats = get_column_statistics(data_df['FP_M2'].astype(float))
    fp_c3_stats = get_column_statistics(data_df['FP_M3'].astype(float))
    
    dsc_stats = [round(d, 2) for d in dsc_stats]
    fpv_stats = [round(d, 2) for d in fpv_stats]
    fnv_stats = [round(d, 2) for d in fnv_stats]
    sens_c1_stats = [round(d, 2) for d in sens_c1_stats]
    sens_c2_stats = [round(d, 2) for d in sens_c2_stats]
    sens_c3_stats = [round(d, 2) for d in sens_c3_stats]
    fp_c1_stats = [round(d, 0) for d in fp_c1_stats]
    fp_c2_stats = [round(d, 0) for d in fp_c2_stats]
    fp_c3_stats = [round(d, 0) for d in fp_c3_stats]

    print(f"DSC (Mean): {dsc_stats[0]} +/- {dsc_stats[1]}")
    print(f"DSC (Median): {dsc_stats[2]} [{dsc_stats[3]}, {dsc_stats[4]}]")
    print(f"FPV (Median): {fpv_stats[2]} [{fpv_stats[3]}, {fpv_stats[4]}]")
    print(f"FNV (Median): {fnv_stats[2]} [{fnv_stats[3]}, {fnv_stats[4]}]")
    print(f"Sensitivity - Criterion1 (Median): {sens_c1_stats[2]} [{sens_c1_stats[3]}, {sens_c1_stats[4]}]")
    print(f"FP - Criterion1 (Median): {fp_c1_stats[2]} [{fp_c1_stats[3]}, {fp_c1_stats[4]}]")
    print(f"Sensitivity - Criterion2 (Median): {sens_c2_stats[2]} [{sens_c2_stats[3]}, {sens_c2_stats[4]}]")
    print(f"FP - Criterion1 (Median): {fp_c2_stats[2]} [{fp_c2_stats[3]}, {fp_c2_stats[4]}]")
    print(f"Sensitivity - Criterion3 (Median): {sens_c3_stats[2]} [{sens_c3_stats[3]}, {sens_c3_stats[4]}]")
    print(f"FP - Criterion3 (Median): {fp_c3_stats[2]} [{fp_c3_stats[3]}, {fp_c3_stats[4]}]")
    print('\n')
    
#%%
def main(args):
    fold = args.fold
    network = args.network_name
    inputsize = args.input_patch_size
    experiment_code = f"{network}_fold{fold}_randcrop{inputsize}"
    preddir = os.path.join(RESULTS_FOLDER, 'predictions', f'fold{fold}', network, experiment_code)
    predpaths = sorted(glob(os.path.join(preddir, '*.nii.gz')))
    gtpaths = sorted(list(pd.read_csv('./../data_split/test_filepaths.csv')['GTPATH']))
    ptpaths = sorted(list(pd.read_csv('./../data_split/test_filepaths.csv')['PTPATH'])) # PET image paths (ptpaths) for calculating the detection metrics using criterion3 
    
    imageids = [os.path.basename(path)[:-7] for path in gtpaths]
    TEST_DSCs, TEST_FPVs, TEST_FNVs = [], [], []
    TEST_TP_criterion1, TEST_FP_criterion1, TEST_FN_criterion1 = [], [], []
    TEST_TP_criterion2, TEST_FP_criterion2, TEST_FN_criterion2 = [], [], []
    TEST_TP_criterion3, TEST_FP_criterion3, TEST_FN_criterion3 = [], [], []

        
    for i in range(len(gtpaths)):
        gtpath = gtpaths[i]
        ptpath = ptpaths[i]
        predpath = predpaths[i]

        gtarray = get_3darray_from_niftipath(gtpath)
        ptarray = get_3darray_from_niftipath(ptpath)
        predarray = get_3darray_from_niftipath(predpath)
        spacing = get_spacing_from_niftipath(gtpath)

        dsc = calculate_patient_level_dice_score(gtarray, predarray)
        fpv = calculate_patient_level_false_positive_volume(gtarray, predarray, spacing)
        fnv = calculate_patient_level_false_negative_volume(gtarray, predarray, spacing)
        tp_c1, fp_c1, fn_c1 = calculate_patient_level_tp_fp_fn(gtarray, predarray, criterion='criterion1')
        tp_c2, fp_c2, fn_c2 = calculate_patient_level_tp_fp_fn(gtarray, predarray, criterion='criterion2', threshold=0.5)
        tp_c3, fp_c3, fn_c3 = calculate_patient_level_tp_fp_fn(gtarray, predarray, criterion='criterion3', ptarray=ptarray)
        
        TEST_DSCs.append(dsc)
        TEST_FPVs.append(fpv)
        TEST_FNVs.append(fnv)
        TEST_TP_criterion1.append(tp_c1)
        TEST_FP_criterion1.append(fp_c1)
        TEST_FN_criterion1.append(fn_c1)
        
        TEST_TP_criterion2.append(tp_c2)
        TEST_FP_criterion2.append(fp_c2)
        TEST_FN_criterion2.append(fn_c2)
        
        TEST_TP_criterion3.append(tp_c3)
        TEST_FP_criterion3.append(fp_c3)
        TEST_FN_criterion3.append(fn_c3)
        print(f"{imageids[i]}: DSC = {round(dsc, 4)}\nFPV = {round(fpv, 4)} ml\nFNV = {round(fnv, 4)} ml")

    save_testmetrics_dir = os.path.join(RESULTS_FOLDER, 'test_metrics', 'fold'+str(fold), network, experiment_code)
    os.makedirs(save_testmetrics_dir, exist_ok=True)
    save_testmetrics_fpath = os.path.join(save_testmetrics_dir, 'testmetrics.csv')

    data = np.column_stack(
        (
            imageids, TEST_DSCs, TEST_FPVs, TEST_FNVs,
            TEST_TP_criterion1, TEST_FP_criterion1, TEST_FN_criterion1,
            TEST_TP_criterion2, TEST_FP_criterion2, TEST_FN_criterion2,
            TEST_TP_criterion3, TEST_FP_criterion3, TEST_FN_criterion3
        )
    )
    column_names = [
        'PatientID', 'DSC', 'FPV', 'FNV',
        'TP_C1', 'FP_C1', 'FN_C1',
        'TP_C2', 'FP_C2', 'FN_C2',
        'TP_C3', 'FP_C3', 'FN_C3',
    ]
    data_df = pd.DataFrame(data=data, columns=column_names)
    data_df.to_csv(save_testmetrics_fpath, index=False)
    
    

    
if __name__ == "__main__":  
    parser = argparse.ArgumentParser(description='Lymphoma PET/CT lesion segmentation using MONAI-PyTorch')
    parser.add_argument('--fold', type=int, default=0, metavar='fold',
                        help='validation fold (default: 0), remaining folds will be used for training')
    parser.add_argument('--network-name', type=str, default='unet', metavar='netname',
                        help='network name for training (default: unet)')
    parser.add_argument('--input-patch-size', type=int, default=192, metavar='inputsize',
                        help='size of cropped input patch for training (default: 192)')
    args = parser.parse_args()
    main(args)
    
# %%
