#%%
'''
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
'''
import pandas as pd 
import os 
from glob import glob 
import numpy as np 
from sklearn.model_selection import train_test_split
# %%
dlbcl_bccv_dir = '/data/blobfuse/lymphoma_lesionsize_split/dlbcl_bccv/all/labels'
pmbcl_bccv_dir = '/data/blobfuse/lymphoma_lesionsize_split/pmbcl_bccv/all/labels'
dlbcl_smhs_dir = '/data/blobfuse/lymphoma_lesionsize_split/dlbcl_smhs/all/labels'

dlbcl_bccv_paths = sorted(glob(os.path.join(dlbcl_bccv_dir, '*.nii.gz')))
pmbcl_bccv_paths = sorted(glob(os.path.join(pmbcl_bccv_dir, '*.nii.gz')))
dlbcl_smhs_paths = sorted(glob(os.path.join(dlbcl_smhs_dir, '*.nii.gz')))

dlbcl_bccv_caseids = [os.path.basename(path)[:-7] for path in dlbcl_bccv_paths]
pmbcl_bccv_caseids = [os.path.basename(path)[:-7] for path in pmbcl_bccv_paths]
dlbcl_smhs_caseids = [os.path.basename(path)[:-7] for path in dlbcl_smhs_paths]
# %%
dlbcl_bccv_ptids = np.unique([path.split('_')[1] for path in dlbcl_bccv_caseids])
pmbcl_bccv_ptids = np.unique([path.split('_')[1] for path in pmbcl_bccv_caseids])
dlbcl_smhs_ptids = np.unique([path.split('_')[1] for path in dlbcl_smhs_caseids])

# %%
def get_caseids_from_ptids(patientids, caseids_all):
    caseids_required = []
    for caseid in caseids_all:
        ptid = caseid.split('_')[1]
        if ptid in patientids:
            caseids_required.append(caseid)
    
    return caseids_required


#%%
# DLBCL (BCCV)
dlbcl_bccv_ptids_train, dlbcl_bccv_ptids_test = train_test_split(dlbcl_bccv_ptids, test_size=0.2, random_state=42)
dlbcl_bccv_ptids_train, dlbcl_bccv_ptids_valid = train_test_split(dlbcl_bccv_ptids_train, test_size=0.2, random_state=42)

dlbcl_bccv_caseids_train = get_caseids_from_ptids(dlbcl_bccv_ptids_train, dlbcl_bccv_caseids)
dlbcl_bccv_caseids_valid = get_caseids_from_ptids(dlbcl_bccv_ptids_valid, dlbcl_bccv_caseids)
dlbcl_bccv_caseids_test = get_caseids_from_ptids(dlbcl_bccv_ptids_test, dlbcl_bccv_caseids)

# %%
# PMBCL (BCCV)
pmbcl_bccv_ptids_train, pmbcl_bccv_ptids_test = train_test_split(pmbcl_bccv_ptids, test_size=0.2, random_state=42)
pmbcl_bccv_ptids_train, pmbcl_bccv_ptids_valid = train_test_split(pmbcl_bccv_ptids_train, test_size=0.2, random_state=42)

pmbcl_bccv_caseids_train = get_caseids_from_ptids(pmbcl_bccv_ptids_train, pmbcl_bccv_caseids)
pmbcl_bccv_caseids_valid = get_caseids_from_ptids(pmbcl_bccv_ptids_valid, pmbcl_bccv_caseids)
pmbcl_bccv_caseids_test = get_caseids_from_ptids(pmbcl_bccv_ptids_test, pmbcl_bccv_caseids)

#%%
# DLBCL (SMHS)
dlbcl_smhs_ptids_train, dlbcl_smhs_ptids_test = train_test_split(dlbcl_smhs_ptids, test_size=0.2, random_state=42)
dlbcl_smhs_ptids_train, dlbcl_smhs_ptids_valid = train_test_split(dlbcl_smhs_ptids_train, test_size=0.2, random_state=42)

dlbcl_smhs_caseids_train = get_caseids_from_ptids(dlbcl_smhs_ptids_train, dlbcl_smhs_caseids)
dlbcl_smhs_caseids_valid = get_caseids_from_ptids(dlbcl_smhs_ptids_valid, dlbcl_smhs_caseids)
dlbcl_smhs_caseids_test = get_caseids_from_ptids(dlbcl_smhs_ptids_test, dlbcl_smhs_caseids)

# %%
# train, valid, test lists
train_list = dlbcl_bccv_caseids_train + pmbcl_bccv_caseids_train + dlbcl_smhs_caseids_train
valid_list = dlbcl_bccv_caseids_valid + pmbcl_bccv_caseids_valid + dlbcl_smhs_caseids_valid
test_list = dlbcl_bccv_caseids_test + pmbcl_bccv_caseids_test + dlbcl_smhs_caseids_test


#%%
# create_train_list
def get_ctpaths_ptpaths_gtpaths_from_caseids(
        caseids_list, 
        dlbcl_bccv_dir: str = '/data/blobfuse/lymphoma_lesionsize_split/dlbcl_bccv/all/', 
        pmbcl_bccv_dir: str = '/data/blobfuse/lymphoma_lesionsize_split/pmbcl_bccv/all/',
        dlbcl_smhs_dir: str = '/data/blobfuse/lymphoma_lesionsize_split/dlbcl_smhs/all/',
) -> (list, list, list):
    ctpaths, ptpaths, gtpaths = [], [], []
    for item in caseids_list:
        studyid = item.split('_')[0]
        if studyid == 'dlbcl-bccv':
            imagesdir = os.path.join(dlbcl_bccv_dir, 'images')
            labelsdir = os.path.join(dlbcl_bccv_dir, 'labels')
        elif studyid == 'pmbcl-bccv':
            imagesdir = os.path.join(pmbcl_bccv_dir, 'images')
            labelsdir = os.path.join(pmbcl_bccv_dir, 'labels')
        elif studyid == 'dlbcl-smhs':
            imagesdir = os.path.join(dlbcl_smhs_dir, 'images')
            labelsdir = os.path.join(dlbcl_smhs_dir, 'labels')
        else:
            pass

        ctpath = os.path.join(imagesdir, f'{item}_0000.nii.gz')
        ptpath = os.path.join(imagesdir, f'{item}_0001.nii.gz')
        gtpath = os.path.join(labelsdir, f'{item}.nii.gz')

        ctpaths.append(ctpath)
        ptpaths.append(ptpath)
        gtpaths.append(gtpath)
    
    return ctpaths, ptpaths, gtpaths

# %%
ctpaths_train, ptpaths_train, gtpaths_train = get_ctpaths_ptpaths_gtpaths_from_caseids(train_list)
ctpaths_valid, ptpaths_valid, gtpaths_valid = get_ctpaths_ptpaths_gtpaths_from_caseids(valid_list)
ctpaths_test, ptpaths_test, gtpaths_test = get_ctpaths_ptpaths_gtpaths_from_caseids(test_list)

# %%
def save_ctpaths_ptpaths_gtpaths_dataframe(ctpaths, ptpaths, gtpaths, savepath):
    data = np.column_stack((ctpaths, ptpaths, gtpaths))
    df = pd.DataFrame(data, columns=['CTPATH', 'PTPATH', 'GTPATH'])
    df.to_csv(savepath, index=False)

# %%
train_filepath = 'train_fold0_patient_level_split.csv'
save_ctpaths_ptpaths_gtpaths_dataframe(ctpaths_train, ptpaths_train, gtpaths_train, train_filepath)

#%%
valid_filepath = 'valid_fold0_patient_level_split.csv'
save_ctpaths_ptpaths_gtpaths_dataframe(ctpaths_valid, ptpaths_valid, gtpaths_valid, valid_filepath)

#%%
test_filepath = 'test_fold0_patient_level_split.csv'
save_ctpaths_ptpaths_gtpaths_dataframe(ctpaths_test, ptpaths_test, gtpaths_test, test_filepath)
# %%
