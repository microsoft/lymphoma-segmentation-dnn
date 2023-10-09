#%%
'''
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
'''
import numpy as np 
import pandas as pd 
import glob 
import os 
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
sys.path.append(PARENT_DIR)
from metrics.metrics import *
# %%
'''
Easy cases in imagedrive_tools4 (n=35)
Hard cases in imagedrive_tools3 (n=25)
'''
easy_or_hard = 'easy'

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
PatientIDs = []
DSC_old_new = []
SUVmean_old, SUVmean_new = [], []
SUVmax_old, SUVmax_new = [], []
LesionCount_old, LesionCount_new = [], []
TMTV_old, TMTV_new = [], []
TLG_old, TLG_new = [], []
Dmax_old, Dmax_new = [], []
#%%
import time 
start = time.time()
for i in range(len(gtpaths_old)):
    ptpath = ptpaths[i]
    gtpath_old = gtpaths_old[i]
    gtpath_new = gtpaths_new[i]

    patientid = os.path.basename(gtpath_old)[:-7]

    ptarray = get_3darray_from_niftipath(ptpath)
    gtarray_old = get_3darray_from_niftipath(gtpath_old)
    gtarray_new = get_3darray_from_niftipath(gtpath_new)

    spacing_old = sitk.ReadImage(gtpath_old).GetSpacing()
    spacing_new = sitk.ReadImage(gtpath_new).GetSpacing()

    # Dice score between mask 1 and 2
    dsc = calculate_patient_level_dice_score(gtarray_old, gtarray_new)

    # Lesion SUVmean 1 and 2
    suvmean_old = calculate_patient_level_lesion_suvmean_suvmax(ptarray, gtarray_old, marker='SUVmean')
    suvmean_new = calculate_patient_level_lesion_suvmean_suvmax(ptarray, gtarray_new, marker='SUVmean')

    # Lesion SUVmax 1 and 2
    suvmax_old = calculate_patient_level_lesion_suvmean_suvmax(ptarray, gtarray_old, marker='SUVmax')
    suvmax_new = calculate_patient_level_lesion_suvmean_suvmax(ptarray, gtarray_new, marker='SUVmax')

    # Lesion Count 1 and 2
    lesioncount_old = calculate_patient_level_lesion_count(gtarray_old)
    lesioncount_new = calculate_patient_level_lesion_count(gtarray_new)

    # TMTV 1 and 2
    tmtv_old = calculate_patient_level_tmtv(gtarray_old, spacing_old)
    tmtv_new = calculate_patient_level_tmtv(gtarray_new, spacing_new)

    # TLG 1 and 2
    tlg_old = calculate_patient_level_tlg(ptarray, gtarray_old, spacing_old)
    tlg_new = calculate_patient_level_tlg(ptarray, gtarray_new, spacing_new)

    # Dmax 1 and 2
    dmax_old = calculate_patient_level_dissemination(gtarray_old, spacing_old)
    dmax_new = calculate_patient_level_dissemination(gtarray_new, spacing_new)

    PatientIDs.append(patientid)
    DSC_old_new.append(dsc)
    SUVmean_old.append(suvmean_old)
    SUVmean_new.append(suvmean_new)
    SUVmax_old.append(suvmax_old)
    SUVmax_new.append(suvmax_new)
    LesionCount_old.append(lesioncount_old)
    LesionCount_new.append(lesioncount_new)
    TMTV_old.append(tmtv_old)
    TMTV_new.append(tmtv_new)
    TLG_old.append(tlg_old)
    TLG_new.append(tlg_new)
    Dmax_old.append(dmax_old)
    Dmax_new.append(dmax_new)

    print(patientid)
    print(f"Dice Score: {round(dsc,4)}")
    print(f"SUVmean: 1: {suvmean_old}, 2: {suvmean_new}")
    print(f"SUVmax: 1: {suvmax_old}, 2: {suvmax_new}")
    print(f"LesionCount: 1: {lesioncount_old}, 2: {lesioncount_new}")
    print(f"TMTV: 1: {tmtv_old}, 2: {tmtv_new}")
    print(f"TLG: 1: {tlg_old}, 2: {tlg_new}")
    print(f"Dmax: 1: {dmax_old}, 2: {dmax_new}")
    print("\n")
# %%
elapsed = time.time() - start
print(elapsed/60)
#%%
data = np.column_stack(
    [
        PatientIDs,
        DSC_old_new,
        SUVmean_old,
        SUVmean_new,
        SUVmax_old,
        SUVmax_new,
        LesionCount_old,
        LesionCount_new,
        TMTV_old,
        TMTV_new,
        TLG_old,
        TLG_new,
        Dmax_old,
        Dmax_new
    ]
)

data_df = pd.DataFrame(
    data=data,
    columns=[
        'PatientID',
        'DSC_old_new',
        'SUVmean_old',
        'SUVmean_new',
        'SUVmax_old',
        'SUVmax_new',
        'LesionCount_old',
        'LesionCount_new',
        'TMTV_old',
        'TMTV_new',
        'TLG_old',
        'TLG_new',
        'Dmax_old',
        'Dmax_new'
    ]
)
#%%
if easy_or_hard == 'easy':
    suffix = '>0p75'
else:
    suffix = '<0p2'
filesavedir = f'{easy_or_hard}_cases_unetdice{suffix}'
data_df.to_csv(os.path.join(filesavedir, f'biomarker_analysis_{easy_or_hard}cases.csv'), index=False)
