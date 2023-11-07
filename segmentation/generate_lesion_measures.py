#%%
'''
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
'''
import pandas as pd 
import numpy as np
import SimpleITK as sitk 
import os 
from glob import glob
import sys
import argparse
config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(config_dir)
from config import RESULTS_FOLDER
from metrics.metrics import *

def get_spacing_from_niftipath(path):
    spacing = sitk.ReadImage(path).GetSpacing()
    return spacing


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
    DSC = [] 
    SUVmean_orig, SUVmean_pred = [], []
    SUVmax_orig, SUVmax_pred = [], [] 
    LesionCount_orig, LesionCount_pred = [], [] 
    TMTV_orig, TMTV_pred = [], []
    TLG_orig, TLG_pred = [], []
    Dmax_orig, Dmax_pred = [], []
    
    for i in range(len(gtpaths)):
        ptpath = ptpaths[i]
        gtpath = gtpaths[i]
        predpath = predpaths[i]
        
        ptarray = get_3darray_from_niftipath(ptpath)
        gtarray = get_3darray_from_niftipath(gtpath)
        predarray = get_3darray_from_niftipath(predpath)
        spacing = get_spacing_from_niftipath(gtpath)

        # Dice score between mask gt and pred
        dsc = calculate_patient_level_dice_score(gtarray, predarray)
        # Lesion SUVmean
        suvmean_orig = calculate_patient_level_lesion_suvmean_suvmax(ptarray, gtarray, marker='SUVmean')
        suvmean_pred = calculate_patient_level_lesion_suvmean_suvmax(ptarray, predarray, marker='SUVmean')
        # Lesion SUVmax
        suvmax_orig = calculate_patient_level_lesion_suvmean_suvmax(ptarray, gtarray, marker='SUVmax')
        suvmax_pred = calculate_patient_level_lesion_suvmean_suvmax(ptarray, predarray, marker='SUVmax')
        # Lesion Count 
        lesioncount_orig = calculate_patient_level_lesion_count(gtarray)
        lesioncount_pred = calculate_patient_level_lesion_count(predarray)
        # TMTV
        tmtv_orig = calculate_patient_level_tmtv(gtarray, spacing)
        tmtv_pred = calculate_patient_level_tmtv(predarray, spacing)
        # TLG
        tlg_orig = calculate_patient_level_tlg(ptarray, gtarray, spacing)
        tlg_pred = calculate_patient_level_tlg(ptarray, predarray, spacing)
        # Dmax
        dmax_orig = calculate_patient_level_dissemination(gtarray, spacing)
        dmax_pred = calculate_patient_level_dissemination(predarray, spacing)
        
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
        
        
        print(f"{i}: {imageids[i]}")
        print(f"Dice Score: {round(dsc,4)}")
        print(f"SUVmean: GT: {suvmean_orig}, Pred: {suvmean_pred}")
        print(f"SUVmax: GT: {suvmax_orig}, Pred: {suvmax_pred}")
        print(f"LesionCount: GT: {lesioncount_orig}, Pred: {lesioncount_pred}")
        print(f"TMTV: GT: {tmtv_orig} ml, Pred: {tmtv_pred} ml")
        print(f"TLG: GT: {tlg_orig} ml, Pred: {tlg_pred} ml")
        print(f"Dmax: GT: {dmax_orig} cm, Pred: {dmax_pred} cm")
        print("\n")

    save_lesionmeasures_dir = os.path.join(RESULTS_FOLDER, f'test_lesion_measures', 'fold'+str(fold), network, experiment_code)
    os.makedirs(save_lesionmeasures_dir, exist_ok=True)
    filepath = os.path.join(save_lesionmeasures_dir, f'testlesionmeasures.csv')
    
    data = np.column_stack(
            [
                imageids,
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
    