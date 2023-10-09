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
import SimpleITK as sitk 
import matplotlib.pyplot as plt 
# import pingouin as pg
# %%
dir = '/data/blobfuse/lymphoma_lesionsize_split/inter_observer_variability_cases'

# Physician1: IB
# Physician2: DW
# Physician3: PM

ibgtdir = os.path.join(dir, 'ib_gt')
dwgtdir = os.path.join(dir, 'dw_gt')
pmgtdir = os.path.join(dir, 'pm_gt')

gtpaths_phys1 = sorted(glob.glob(os.path.join(ibgtdir, '*.nii.gz')))
gtpaths_phys2 = sorted(glob.glob(os.path.join(dwgtdir, '*.nii.gz')))
gtpaths_phys3 = sorted(glob.glob(os.path.join(pmgtdir, '*.nii.gz')))
PatientIDs = [f"dlbcl-bccv_{os.path.basename(path)[:-13]}" for path in gtpaths_phys1]
#%%
dir = '/data/blobfuse/lymphoma_lesionsize_split/dlbcl_bccv/all/'
ptdir = os.path.join(dir, 'images')
ptpaths = []
for i in range(len(PatientIDs)):
    ptpath = os.path.join(ptdir, f"{PatientIDs[i]}_0001.nii.gz")
    ptpaths.append(ptpath)

# %%
SUVmean_phys1, SUVmean_phys2, SUVmean_phys3 = [], [], []
SUVmax_phys1, SUVmax_phys2, SUVmax_phys3 = [], [], []
LesionCount_phys1, LesionCount_phys2, LesionCount_phys3 =  [], [], []
TMTV_phys1, TMTV_phys2, TMTV_phys3=  [], [], []
TLG_phys1, TLG_phys2, TLG_phys3 =  [], [], []
Dmax_phys1, Dmax_phys2, Dmax_phys3=  [], [], []
#%%
import time 
start = time.time()
for i in range(len(PatientIDs)):
    ptpath = ptpaths[i]
    gtpath_phys1 = gtpaths_phys1[i]
    gtpath_phys2 = gtpaths_phys2[i]
    gtpath_phys3 = gtpaths_phys3[i]

    ptarray = get_3darray_from_niftipath(ptpath)
    gtarray_phys1 = get_3darray_from_niftipath(gtpath_phys1)
    gtarray_phys2 = get_3darray_from_niftipath(gtpath_phys2)
    gtarray_phys3 = get_3darray_from_niftipath(gtpath_phys3)

    spacing_phys1 = sitk.ReadImage(gtpath_phys1).GetSpacing()
    spacing_phys2 = sitk.ReadImage(gtpath_phys2).GetSpacing()
    spacing_phys3 = sitk.ReadImage(gtpath_phys3).GetSpacing()

    # Dice score between mask 1 and 2

    # Lesion SUVmean 1 and 2
    suvmean_phys1 = calculate_patient_level_lesion_suvmean_suvmax(ptarray, gtarray_phys1, marker='SUVmean')
    suvmean_phys2 = calculate_patient_level_lesion_suvmean_suvmax(ptarray, gtarray_phys2, marker='SUVmean')
    suvmean_phys3 = calculate_patient_level_lesion_suvmean_suvmax(ptarray, gtarray_phys3, marker='SUVmean')

    # Lesion SUVmax 1 and 2
    suvmax_phys1 = calculate_patient_level_lesion_suvmean_suvmax(ptarray, gtarray_phys1, marker='SUVmax')
    suvmax_phys2 = calculate_patient_level_lesion_suvmean_suvmax(ptarray, gtarray_phys2, marker='SUVmax')
    suvmax_phys3 = calculate_patient_level_lesion_suvmean_suvmax(ptarray, gtarray_phys3, marker='SUVmax')

    # Lesion Count 1 and 2
    lesioncount_phys1 = calculate_patient_level_lesion_count(gtarray_phys1)
    lesioncount_phys2 = calculate_patient_level_lesion_count(gtarray_phys2)
    lesioncount_phys3 = calculate_patient_level_lesion_count(gtarray_phys3)

    # TMTV 1 and 2
    tmtv_phys1 = calculate_patient_level_tmtv(gtarray_phys1, spacing_phys1)
    tmtv_phys2 = calculate_patient_level_tmtv(gtarray_phys2, spacing_phys2)
    tmtv_phys3 = calculate_patient_level_tmtv(gtarray_phys3, spacing_phys3)

    # TLG 1 and 2
    tlg_phys1 = calculate_patient_level_tlg(ptarray, gtarray_phys1, spacing_phys1)
    tlg_phys2 = calculate_patient_level_tlg(ptarray, gtarray_phys2, spacing_phys2)
    tlg_phys3 = calculate_patient_level_tlg(ptarray, gtarray_phys3, spacing_phys3)

    # Dmax 1 and 2
    dmax_phys1 = calculate_patient_level_dissemination(gtarray_phys1, spacing_phys1)
    dmax_phys2 = calculate_patient_level_dissemination(gtarray_phys2, spacing_phys2)
    dmax_phys3 = calculate_patient_level_dissemination(gtarray_phys3, spacing_phys3)
    
    SUVmean_phys1.append(suvmean_phys1)
    SUVmean_phys2.append(suvmean_phys2)
    SUVmean_phys3.append(suvmean_phys3)

    SUVmax_phys1.append(suvmax_phys1)
    SUVmax_phys2.append(suvmax_phys2)
    SUVmax_phys3.append(suvmax_phys3)

    LesionCount_phys1.append(lesioncount_phys1)
    LesionCount_phys2.append(lesioncount_phys2)
    LesionCount_phys3.append(lesioncount_phys3)

    TMTV_phys1.append(tmtv_phys1)
    TMTV_phys2.append(tmtv_phys2)
    TMTV_phys3.append(tmtv_phys3)

    TLG_phys1.append(tlg_phys1)
    TLG_phys2.append(tlg_phys2)
    TLG_phys3.append(tlg_phys3)

    Dmax_phys1.append(dmax_phys1)
    Dmax_phys2.append(dmax_phys2)
    Dmax_phys3.append(dmax_phys3)

    print(PatientIDs[i])
    print(f"SUVmean: 1: {suvmean_phys1}, 2: {suvmean_phys2}, 3: {suvmean_phys3}")
    print(f"SUVmax: 1: {suvmax_phys1}, 2: {suvmax_phys2}, 3: {suvmax_phys3}")
    print(f"LesionCount: 1: {lesioncount_phys1}, 2: {lesioncount_phys2}, 3: {lesioncount_phys3}")
    print(f"TMTV: 1: {tmtv_phys1}, 2: {tmtv_phys2}, 3: {tmtv_phys3}")
    print(f"TLG: 1: {tlg_phys1}, 2: {tlg_phys2}, 3: {tlg_phys3}")
    print(f"Dmax: 1: {dmax_phys1}, 2: {dmax_phys2}, 3: {dmax_phys3}")
    print("\n")
# %%
elapsed = time.time() - start
print(elapsed/60)
#%%
data = np.column_stack(
    [
        PatientIDs,
        SUVmean_phys1,
        SUVmean_phys2,
        SUVmean_phys3,
        SUVmax_phys1,
        SUVmax_phys2,
        SUVmax_phys3,
        LesionCount_phys1,
        LesionCount_phys2,
        LesionCount_phys3,
        TMTV_phys1,
        TMTV_phys2,
        TMTV_phys3,
        TLG_phys1,
        TLG_phys2,
        TLG_phys3,
        Dmax_phys1,
        Dmax_phys2,
        Dmax_phys3
    ]
)

data_df = pd.DataFrame(
    data=data,
    columns=[
        'PatientID',
        'SUVmean_phys1',
        'SUVmean_phys2',
        'SUVmean_phys3',
        'SUVmax_phys1',
        'SUVmax_phys2',
        'SUVmax_phys3',
        'LesionCount_phys1',
        'LesionCount_phys2',
        'LesionCount_phys3',
        'TMTV_phys1',
        'TMTV_phys2',
        'TMTV_phys3',
        'TLG_phys1',
        'TLG_phys2',
        'TLG_phys3',
        'Dmax_phys1',
        'Dmax_phys2',
        'Dmax_phys3'
    ]
)
#%%
data_df.to_csv('inter_obs_biomarker_analysis.csv', index=False)

#%%
data_df = pd.read_csv('inter_obs_biomarker_analysis.csv')

#%%

def get_interclass_corr(data_df, biomarker='SUVmean'):
    biomarker_phys1 = data_df[[f'PatientID', f'{biomarker}_phys1']]
    biomarker_phys2 = data_df[[f'PatientID', f'{biomarker}_phys2']]
    biomarker_phys3 = data_df[[f'PatientID', f'{biomarker}_phys3']]

    biomarker_phys1 = biomarker_phys1.astype({f'{biomarker}_phys1': float})
    biomarker_phys2 = biomarker_phys2.astype({f'{biomarker}_phys2': float})
    biomarker_phys3 = biomarker_phys3.astype({f'{biomarker}_phys3': float})

    biomarker_phys1 = biomarker_phys1.rename(columns={f'{biomarker}_phys1': f'{biomarker}'})
    biomarker_phys2 = biomarker_phys2.rename(columns={f'{biomarker}_phys2': f'{biomarker}'})
    biomarker_phys3 = biomarker_phys3.rename(columns={f'{biomarker}_phys3': f'{biomarker}'})
    
    annotator_df_phys1 = pd.DataFrame(['Phys1']*len(biomarker_phys1), columns=['Rater'])
    annotator_df_phys2 = pd.DataFrame(['Phys2']*len(biomarker_phys2), columns=['Rater'])
    annotator_df_phys3 = pd.DataFrame(['Phys3']*len(biomarker_phys3), columns=['Rater'])

    biomarker_phys1_df = pd.concat([biomarker_phys1, annotator_df_phys1], axis=1)
    biomarker_phys2_df = pd.concat([biomarker_phys2, annotator_df_phys2], axis=1)
    biomarker_phys3_df = pd.concat([biomarker_phys3, annotator_df_phys3], axis=1)

    biomarker_df = pd.concat([biomarker_phys1_df, biomarker_phys2_df, biomarker_phys3_df], axis=0)
    icc = pg.intraclass_corr(data=biomarker_df, targets='PatientID', raters='Rater', ratings=biomarker).round(2)
    # return icc
    icc3k = icc[icc['Type'] == 'ICC3k']
    return (icc3k['ICC'].values[0], list(icc3k['CI95%'].values[0])) 

def plot_biomarkes_of_three_physicians(data_df, ax = None, biomarker='SUVmean', title=r'SUV$_{mean}$', unit='', j=0):
    biomarker_phys1_phys2_phys3 = data_df[[f'{biomarker}_phys1', f'{biomarker}_phys2', f'{biomarker}_phys3']].astype(float)
    biomarker_df = biomarker_phys1_phys2_phys3.dropna()
    phys1_values = biomarker_df.iloc[:,0].values
    phys2_values = biomarker_df.iloc[:,1].values
    phys3_values = biomarker_df.iloc[:,2].values
   
    x_axis = np.arange(1,len(phys1_values)+1)
    ax.scatter(x_axis, phys1_values, marker='o', s=300, color='cyan')
    ax.scatter(x_axis, phys2_values, marker='s', s=100, color='magenta')
    ax.scatter(x_axis, phys3_values, marker='x', s=200, color='green', linewidth=3)
    

    (icc3k_val, icc3k_ci95)  = get_interclass_corr(data_df, biomarker=biomarker)
    icc3k_val_formatted = "{:.2f}".format(icc3k_val)
    icc3k_ci95_0_formatted = "{:.2f}".format(icc3k_ci95[0])
    icc3k_ci95_1_formatted = "{:.2f}".format(icc3k_ci95[1])
    text_loc = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[]}
    text_loc_x = np.min(ax.get_xlim()) + 0.15*(np.max(ax.get_xlim()) - np.min(ax.get_xlim()))
    text_loc_y = np.min(ax.get_ylim()) - 0.26*(np.max(ax.get_ylim()) - np.min(ax.get_ylim()))
    if j == 0:
        ax.legend(['Physician 1', 'Physician 2', 'Physician 3'], loc='lower right', bbox_to_anchor=(4.8, -0.5), fontsize=24, ncol=3)
    else: 
        pass
    ax.text(text_loc_x, text_loc_y, f'ICC: {icc3k_val_formatted} ({icc3k_ci95_0_formatted},{icc3k_ci95_1_formatted})', fontsize=20)
    return ax
   


# %%
rows, cols = 1, 6
fig, ax = plt.subplots(1, cols, figsize=(35, 5))
fig.patch.set_facecolor('white')
fig.patch.set_alpha(1)
biomarkers = ['SUVmean', 'SUVmax', 'LesionCount', 'TMTV', 'TLG', 'Dmax']
titles = [r'SUV$_{mean}$', r'SUV$_{max}$', f'Number of lesions', r'TMTV (cm$^3$)', r'TLG (cm$^3$)', r'$D_{max}$ (cm)']

# for i in range(1):
for j in range(6):
    # plot correlations
    plot_biomarkes_of_three_physicians(data_df, ax=ax[j], biomarker=biomarkers[j], title=titles[j], j=j)
    ax[j].set_xlabel('Cases', fontsize=25)
    ax[j].set_title(titles[j], fontsize=30)
    ax[j].tick_params(axis='both', labelsize=15)


fig.savefig('inter_observer_variability_biomarker_agreement.png', dpi=500, bbox_inches='tight')

# %%
