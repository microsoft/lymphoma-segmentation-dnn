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
from metrics.metrics import calculate_patient_level_dice_score
from data_analysis.visualization.plot_pet_mask import plot_image_mask_superposed_coronalmip
import SimpleITK as sitk 
from scipy.ndimage import zoom
#%%
testfpath = '/home/shadab/Projects/lymphoma-segmentation/create_data_split/patient_level_split/test_fold0_patient_level_split.csv'
testdf = pd.read_csv(testfpath)
ptpaths = sorted(list(testdf['PTPATH'].values))
gtpaths = sorted(list(testdf['GTPATH'].values))

preddirs = [
    '/data/blobfuse/lymphoma-segmentation-results-new/test_predictions/fold0/unet/unet_fold0_randcrop224_p2_wd1em5_lr2em4',
    '/data/blobfuse/lymphoma-segmentation-results-new/test_predictions/fold0/segresnet/segresnet_fold0_randcrop192_p2_wd1em5_lr2em4',
    '/data/blobfuse/lymphoma-segmentation-results-new/test_predictions/fold0/dynunet/dynunet_fold0_randcrop160_p2_wd1em5_lr2em4',
    '/data/blobfuse/lymphoma-segmentation-results-new/test_predictions/fold0/swinunetr/swinunetr_fold0_randcrop128_p2_wd1em5_lr2em4'
]

predpaths = [sorted(glob(os.path.join(dir, '*.nii.gz'))) for dir in preddirs]
allpaths = np.column_stack(
    (   ptpaths,
        gtpaths,
        predpaths[0],
        predpaths[1],
        predpaths[2],
        predpaths[3]
    )
)
savedir = '/home/shadab/Projects/lymphoma-segmentation/data_analysis/good_and_bad_network_segmentations'


def get_required_paths(pathdata, imageid):
    for i in range(len(pathdata)):
        if os.path.basename(pathdata[i][1])[:-7] == imageid:
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
    orders = [3, 0, 0, 0, 0, 0]
    arrays_resampled = [resample_array_to_match_template(arrays[i], template_arrays[i], orders[i]) for i in range(len(arrays))]
    return arrays_resampled


def get_arrays_from_images(sitk_images):
    arrays = [np.transpose(sitk.GetArrayFromImage(image), (2,1,0)) for image in sitk_images]
    return arrays
    
#%%
titles = [
    'Ground truth',
    'UNet',
    'SegResNet',
    'DynUNet',
    'SwinUNETR'
]

# I want to keep this list `paper_plots_images`
# paper_plots_images = [
#     'pmbcl-bccv_16-25535_20170822', # similar 
#     'pmbcl-bccv_14-31910_20150205', # 1243
#     'pmbcl-bccv_09-39732_20100601', # 1234
#     'dlbcl-smhs_881077517_20180723', # 2431
#     'dlbcl-smhs_81067111_20181109', # 4123
#     'dlbcl-smhs_760465304_20170609', # 1342
#     'dlbcl-smhs_610477236_20190712', # 4123
#     'dlbcl-smhs_51772786_20160425', # similar
#     'dlbcl-smhs_464118928_20171103', # similar
#     'dlbcl-smhs_420430641_20200409', # similar
#     'dlbcl-smhs_417043777_20200220', # 1423
#     'dlbcl-smhs_312264457_20190819', # similar
#     'dlbcl-smhs_219034353_20171027', # similar
# ]

images_similar = [
    'dlbcl-smhs_219034353_20171027', # similar,
    # 'dlbcl-smhs_312264457_20190819', # similar
    'dlbcl-smhs_464118928_20171103', # similar
    'dlbcl-smhs_420430641_20200409', # similar
    # 'pmbcl-bccv_16-25535_20170822', # similar 
    'dlbcl-smhs_51772786_20160425' # similar

]
images_different = [
    'pmbcl-bccv_09-39732_20100601', # 1234
    'pmbcl-bccv_14-31910_20150205', # 1243
    
    # 'dlbcl-smhs_417043777_20200220', # 1423
    'dlbcl-smhs_81067111_20181109', # 4123
    'dlbcl-smhs_610477236_20190712', # 4123
]

images_to_plot = images_different
fig, ax = plt.subplots(len(images_to_plot), 5, figsize=(8, 12), gridspec_kw = {'wspace':0, 'hspace':0.00}, constrained_layout=True)
fig.patch.set_facecolor('white')
fig.patch.set_alpha(1)

for i in range(len(images_similar[0:1])):
    imageid = images_similar[i]
    pathdata = get_required_paths(allpaths, imageid)
    sitk_images = get_sitk_images(pathdata)
    arrays = get_arrays_from_images(sitk_images)
    TEMPLATE_ARRAYS = arrays
    


for i in range(len(images_to_plot)):
    imageid = images_to_plot[i]
    pathdata = get_required_paths(allpaths, imageid)
    sitk_images = get_sitk_images(pathdata)
    arrays = get_arrays_from_images(sitk_images)
    dscs = []
    for j in range(2, 6):
        dscs.append(calculate_patient_level_dice_score(arrays[1], arrays[j]))
    print(dscs)

    arrays = get_resampled_arrays(arrays, TEMPLATE_ARRAYS)

    for j in range(1, 6):
        plot_image_mask_superposed_coronalmip(arrays[0], arrays[j], ax=ax[i][j-1])
        if i == 0:
            ax[i][j-1].set_title(titles[j-1], fontsize=15)
    for j in range(2, 6):
        ax[i][j-1].text(np.max(arrays[0].shape[1])*0.65, np.max(arrays[0].shape[2])*0.95, f"{dscs[j-2]:.2f}", fontsize=14)
    
plt.tight_layout()
plt.show()
#%%
fig.savefig('images_dissimilar_performance.png', dpi=400, bbox_inches='tight')
