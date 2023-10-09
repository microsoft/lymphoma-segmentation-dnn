#%%
'''
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
'''
import pandas as pd 
import numpy as np
import SimpleITK as sitk 
import os 
import glob 
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
sys.path.append(PARENT_DIR)
from metrics.metrics import *
import matplotlib.pyplot as plt 
from config import RESULTS_FOLDER
# %%
start_index = 71
autopet_dir = '/data/blobfuse/autopet2022_data'
images_dir = os.path.join(autopet_dir, 'images')
labels_dir = os.path.join(autopet_dir, 'labels')
autopet_lymphoma_metadata_file = './../patient_metadata/autopet_lymphoma.csv'
PatientIDs = list(pd.read_csv(autopet_lymphoma_metadata_file)['PatientID'].values)[start_index:]

ptpaths, gtpaths = [], []
for id in PatientIDs:
    ptpath = os.path.join(images_dir, f'{id}_0001.nii.gz')
    gtpath = os.path.join(labels_dir, f'{id}.nii.gz')
    ptpaths.append(ptpath)
    gtpaths.append(gtpath)

ptpaths = sorted(ptpaths)
gtpaths = sorted(gtpaths)
#%%
fold = 0
network = 'dynunet'
p = 2
extra_features = f'p{p}_wd1em5_lr2em4'
N_for_network = {
    'unet': 224,
    'segresnet': 192,
    'dynunet': 160,
    'swinunetr': 128
}

inputsize = N_for_network[network]
experiment_code = f"{network}_fold{fold}_randcrop{inputsize}_{extra_features}"

save_preds_dir = os.path.join(RESULTS_FOLDER, f'autopet_test_predictions')
save_preds_dir = os.path.join(save_preds_dir, 'fold'+str(fold), network, experiment_code)
os.makedirs(save_preds_dir, exist_ok=True)
predpaths = sorted(glob.glob(os.path.join(save_preds_dir, '*.nii.gz')))[start_index:]


# %%
save_biomakers_dir = os.path.join(RESULTS_FOLDER, f'autopet_biomarkers')
save_biomakers_dir = os.path.join(save_biomakers_dir, 'fold'+str(fold), network, experiment_code)
os.makedirs(save_biomakers_dir, exist_ok=True)
filepath = os.path.join(save_biomakers_dir, f'biomarkers.csv')

if os.path.exists(filepath):
    data_df = pd.read_csv(filepath)
    cols=[
        'PatientID',
        'DSC',
        'SUVmean_orig',
        'SUVmean_pred',
        'SUVmax_orig',
        'SUVmax_pred',
        'LesionCount_orig',
        'LesionCount_pred',
        'TMTV_orig',
        'TMTV_pred',
        'TLG_orig',
        'TLG_pred',
        'Dmax_orig',
        'Dmax_pred'
    ]

    PatientIDs_new, DSC, SUVmean_orig, SUVmean_pred, SUVmax_orig, SUVmax_pred, LesionCount_orig, LesionCount_pred, TMTV_orig, TMTV_pred, TLG_orig, TLG_pred, Dmax_orig, Dmax_pred = [list(data_df[col].values) for col in cols]

else:
    PatientIDs_new, DSC, SUVmean_orig, SUVmean_pred, SUVmax_orig, SUVmax_pred, LesionCount_orig, LesionCount_pred, TMTV_orig, TMTV_pred, TLG_orig, TLG_pred, Dmax_orig, Dmax_pred = [], [],[],[],[],[],[],[],[],[],[],[],[],[],
#%%
import time 
start = time.time()
missed_cases_orig = []
missed_cases_pred = []



for i in range(len(gtpaths)):
    ptpath = ptpaths[i]
    gtpath = gtpaths[i]
    predpath = predpaths[i]
    patientid = os.path.basename(gtpath)[:-7]
    
    ptarray = get_3darray_from_niftipath(ptpath)
    gtarray = get_3darray_from_niftipath(gtpath)
    predarray = get_3darray_from_niftipath(predpath)
    
    spacing = sitk.ReadImage(gtpath).GetSpacing()

    # Dice score between mask 1 and 2
    dsc = calculate_patient_level_dice_score(gtarray, predarray)

    # Lesion SUVmean 1 and 2
    suvmean_orig = calculate_patient_level_lesion_suvmean_suvmax(ptarray, gtarray, marker='SUVmean')
    suvmean_pred = calculate_patient_level_lesion_suvmean_suvmax(ptarray, predarray, marker='SUVmean')

    # Lesion SUVmax 1 and 2
    suvmax_orig = calculate_patient_level_lesion_suvmean_suvmax(ptarray, gtarray, marker='SUVmax')
    suvmax_pred = calculate_patient_level_lesion_suvmean_suvmax(ptarray, predarray, marker='SUVmax')

    # Lesion Count 1 and 2
    lesioncount_orig = calculate_patient_level_lesion_count(gtarray)
    lesioncount_pred = calculate_patient_level_lesion_count(predarray)

    # TMTV 1 and 2
    tmtv_orig = calculate_patient_level_tmtv(gtarray, spacing)
    tmtv_pred = calculate_patient_level_tmtv(predarray, spacing)

    # TLG 1 and 2
    tlg_orig = calculate_patient_level_tlg(ptarray, gtarray, spacing)
    tlg_pred = calculate_patient_level_tlg(ptarray, predarray, spacing)

    # Dmax 1 and 2
    try:
        dmax_orig = calculate_patient_level_dissemination(gtarray, spacing)
    except:
        dmax_orig = np.nan
        missed_cases_orig.append(PatientIDs[i])
        print(f'Orig: {patientid}: Out-of-memory')
    try:
        dmax_pred = calculate_patient_level_dissemination(predarray, spacing)
    except:
        dmax_pred = np.nan
        missed_cases_pred.append(patientid)
        print(f'Pred: {patientid}: Out-of-memory')

    PatientIDs_new.append(patientid)
    DSC.append(dsc)
    SUVmean_orig.append(suvmean_orig)
    SUVmean_pred.append(suvmean_pred)
    SUVmax_orig.append(suvmax_orig)
    SUVmax_pred.append(suvmax_pred)
    LesionCount_orig.append(lesioncount_orig)
    LesionCount_pred.append(lesioncount_pred)
    TMTV_orig.append(tmtv_orig)
    TMTV_pred.append(tmtv_pred)
    TLG_orig.append(tlg_orig)
    TLG_pred.append(tlg_pred)
    Dmax_orig.append(dmax_orig)
    Dmax_pred.append(dmax_pred)

    print(f"{i}: {patientid}")
    print(f"Dice Score: {round(dsc,4)}")
    print(f"SUVmean: 1: {suvmean_orig}, 2: {suvmean_pred}")
    print(f"SUVmax: 1: {suvmax_orig}, 2: {suvmax_pred}")
    print(f"LesionCount: 1: {lesioncount_orig}, 2: {lesioncount_pred}")
    print(f"TMTV: 1: {tmtv_orig}, 2: {tmtv_pred}")
    print(f"TLG: 1: {tlg_orig}, 2: {tlg_pred}")
    print(f"Dmax: 1: {dmax_orig}, 2: {dmax_pred}")
    print("\n")

    data = np.column_stack(
    [
        PatientIDs_new,
        DSC,
        SUVmean_orig,
        SUVmean_pred,
        SUVmax_orig,
        SUVmax_pred,
        LesionCount_orig,
        LesionCount_pred,
        TMTV_orig,
        TMTV_pred,
        TLG_orig,
        TLG_pred,
        Dmax_orig,
        Dmax_pred
    ]
    )

    data_df = pd.DataFrame(
        data=data,
        columns=[
            'PatientID',
            'DSC',
            'SUVmean_orig',
            'SUVmean_pred',
            'SUVmax_orig',
            'SUVmax_pred',
            'LesionCount_orig',
            'LesionCount_pred',
            'TMTV_orig',
            'TMTV_pred',
            'TLG_orig',
            'TLG_pred',
            'Dmax_orig',
            'Dmax_pred'
        ]
    )
    data_df.to_csv(filepath, index=False)

# %%
elapsed = time.time() - start
print(f"{elapsed/(60*60)} hrs")
#%%
missed_orig_data = pd.DataFrame(missed_cases_orig, columns=['MissedCase'])
missed_pred_data = pd.DataFrame(missed_cases_pred, columns=['MissedCase'])
missed_orig_data.to_csv(f'{network}_autopet_missed_orig.csv', index=False)
missed_pred_data.to_csv(f'{network}_autopet_missed_pred.csv', index=False)

