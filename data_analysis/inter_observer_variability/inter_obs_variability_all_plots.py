#%%
'''
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
'''
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import os 
from glob import glob
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
sys.path.append(PARENT_DIR)
from data_analysis.visualization.plot_pet_mask import plot_image_mask_superposed_coronalmip
import SimpleITK as sitk 
from scipy.ndimage import zoom
#%%
dir = '/data/blobfuse/lymphoma_lesionsize_split/inter_observer_variability_cases'

# Physician1: IB
# Physician2: DW
# Physician3: PM

ibgtdir = os.path.join(dir, 'ib_gt')
dwgtdir = os.path.join(dir, 'dw_gt')
pmgtdir = os.path.join(dir, 'pm_gt')
stapledir = os.path.join(dir, 'staple_agreement')

gtpaths_phys1 = sorted(glob(os.path.join(ibgtdir, '*.nii.gz')))
gtpaths_phys2 = sorted(glob(os.path.join(dwgtdir, '*.nii.gz')))
gtpaths_phys3 = sorted(glob(os.path.join(pmgtdir, '*.nii.gz')))
gtpaths_staple = sorted(glob(os.path.join(stapledir, '*.nii.gz')))

PatientIDs = [f"dlbcl-bccv_{os.path.basename(path)[:-13]}" for path in gtpaths_phys1]

dir = '/data/blobfuse/lymphoma_lesionsize_split/dlbcl_bccv/all/'
ptdir = os.path.join(dir, 'images')
ptpaths = []
for i in range(len(PatientIDs)):
    ptpath = os.path.join(ptdir, f"{PatientIDs[i]}_0001.nii.gz")
    ptpaths.append(ptpath)

#%%

def get_required_paths(pathdata, imageid):
    for i in range(len(pathdata)):
        if os.path.basename(pathdata[i][0])[:-12] == imageid:
            pathdata_required = pathdata[i]
            break
    return pathdata_required

def get_sitk_images(pathdata):
    sitk_images = []
    for path in pathdata:
        image = sitk.ReadImage(path)
        sitk_images.append(image)
    return sitk_images


def resample_array_to_match_template(input_array, template_array, order=0):
    input_shape = input_array.shape
    template_shape = template_array.shape
    zoom_factors = [t / i for t, i in zip(template_shape, input_shape)]
    resampled_array = zoom(input_array, zoom_factors, order=order)
    return resampled_array

def get_resampled_arrays(arrays, template_arrays):
    orders = [3, 0, 0, 0, 0]
    arrays_resampled = [resample_array_to_match_template(arrays[i], template_arrays[i], orders[i]) for i in range(len(arrays))]
    return arrays_resampled


def get_arrays_from_images(sitk_images):
    arrays = [np.transpose(sitk.GetArrayFromImage(image), (2,1,0)) for image in sitk_images]
    return arrays
    
#%%
allpaths = np.column_stack((ptpaths, gtpaths_phys1, gtpaths_phys2, gtpaths_phys3, gtpaths_staple))
#%%
titles = [
    'Physician 1',
    'Physician 2',
    'Physician 3',
    'STAPLE'
]


images_to_plot = [
   'dlbcl-bccv_00-27436_20110502',
   'dlbcl-bccv_01-11153_20150514',
   'dlbcl-bccv_03-11701_20110530',
   'dlbcl-bccv_16-13501_20160311'
]



fig, ax = plt.subplots(len(images_to_plot), 4, figsize=(9, 11))
fig.patch.set_facecolor('white')
fig.patch.set_alpha(1)

for i in range(len(images_to_plot[0:1])):
    imageid = images_to_plot[i]
    pathdata = get_required_paths(allpaths, imageid)
    sitk_images = get_sitk_images(pathdata)
    arrays = get_arrays_from_images(sitk_images)
    TEMPLATE_ARRAYS = arrays

caseids = ['Case 1', 'Case 2', 'Case 3', 'Case 8']
kt = [r'$\kappa$ = 0.41', r'$\kappa$ = 0.98', r'$\kappa$ = 0.38', r'$\kappa$ = 0.95']
labels = [f"{caseids[i]}\n{kt[i]}" for i in range(len(caseids))]
for i in range(len(images_to_plot)):
    imageid = images_to_plot[i]
    pathdata = get_required_paths(allpaths, imageid)
    sitk_images = get_sitk_images(pathdata)
    arrays = get_arrays_from_images(sitk_images)
    
    arrays = get_resampled_arrays(arrays, TEMPLATE_ARRAYS)

    for j in range(1, 5):
        plot_image_mask_superposed_coronalmip(arrays[0], arrays[j], ax=ax[i][j-1])
        if i == 0:
            ax[i][j-1].set_title(titles[j-1], fontsize=15)
        if j-1 == 0:
            ax[i][j-1].text(10, np.max(arrays[0].shape[2])*0.95, f"{labels[i]}", fontsize=14)
    
plt.tight_layout()
plt.show()
#%%
fig.savefig('inter_observer_variability.png', dpi=400, bbox_inches='tight')