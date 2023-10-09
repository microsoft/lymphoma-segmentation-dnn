'''
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
'''

'''
This code does the following:
(1) converts PET DICOM images in units Bq/ml to decay-corrected SUV and saved as 3D NIFTI files
(2) converts CT DICOM images to NIFTI
(3) converts DICOM RTSTRUCT images to NIFTI (using rt-utils)
'''
#%%
import SimpleITK as sitk
from pydicom import dcmread, FileDataset
from rt_utils import RTStructBuilder, RTStruct
import numpy as np
import dateutil
import pandas as pd
import os
import time

#%%
'''
Script to convert PET and CT dicom series to niftii files. Works under 
the assumption that the rescale slope and intercept in the PET dicom 
series map image intensities to Bq/mL. Saved PET files will have image
intensities of SUVbw, and saved CT files will have HU units.

'''
def bqml_to_suv(dcm_file: FileDataset) -> float:
    
    # Calculates the SUV conversion factor from Bq/mL to SUVbw using 
    # the dicom header information in one of the images from a dicom series
    
    nuclide_dose = dcm_file[0x054, 0x0016][0][0x0018, 0x1074].value  # Total injected dose (Bq)
    weight = dcm_file[0x0010, 0x1030].value  # Patient weight (Kg)
    half_life = float(dcm_file[0x054, 0x0016][0][0x0018, 0x1075].value)  # Radionuclide half life (s)

    parse = lambda x: dateutil.parser.parse(x)

    series_time = str(dcm_file[0x0008, 0x00031].value)  # Series start time (hh:mm:ss)
    series_date = str(dcm_file[0x0008, 0x00021].value)  # Series start date (yyy:mm:dd)
    series_datetime_str = series_date + ' ' + series_time
    series_dt = parse(series_datetime_str)

    nuclide_time = str(dcm_file[0x054, 0x0016][0][0x0018, 0x1072].value)  # Radionuclide time of injection (hh:mm:ss)
    nuclide_datetime_str = series_date + ' ' + nuclide_time
    nuclide_dt = parse(nuclide_datetime_str)

    delta_time = (series_dt - nuclide_dt).total_seconds()
    decay_correction = 2 ** (-1 * delta_time/half_life)
    suv_factor = (weight * 1000) / (decay_correction * nuclide_dose)

    return(suv_factor)

def get_filtered_roi_list(rois):
    filtered_rois = []
    for roi in rois:
        if roi.endswith('PETEdge'):
            filtered_rois.append(roi)
        else:
            pass
    return filtered_rois


def load_merge_masks(rtstruct: RTStruct) -> np.ndarray:
    '''
    Load and merge masks from a dicom RTStruct. All of the
    masks in the RTStruct will be merged. Add an extra line
    of code if you want to filter for/out certain masks.
    '''
    rois = rtstruct.get_roi_names()
    rois = get_filtered_roi_list(rois)
    masks = []
    for roi in rois:
        print(roi)
        mask_3d = rtstruct.get_roi_mask_by_name(roi).astype(int)
        masks.append(mask_3d)

    final_mask = sum(masks)  # sums element-wise
    final_mask = np.where(final_mask>=1, 1, 0)
    # Reorient the mask to line up with the reference image
    final_mask = np.moveaxis(final_mask, [0, 1, 2], [1, 2, 0])

    return final_mask

############################################################################################
########  Update the three variables below with the locations of your choice  ##############
############################################################################################
save_dir_ct = '' # path to directory where your new CT files in NIFTI format will be written
save_dir_pt = '' # path to directory where your new PET files in NIFTI format will be written
save_dir_gt = '' # path to directory where your new GT files in NIFTI format will be written
############################################################################################
############################################################################################
############################################################################################

cases = pd.read_csv('dicom_ctpt_to_nifti_conversion_file.csv')
cases = list(cases.itertuples(index=False, name=None)) 
structs = pd.read_csv('dicom_rtstruct_to_nifti_conversion_file.csv')
structs = list(structs.itertuples(index=False, name=None))
# Execution
start = time.time()

for case in cases:
    patient_id, ct_folder, pet_folder, convert = case
    if convert=='N':
        continue
    print(f'Converting patient Id: {patient_id}')

    # Convert CT series
    ct_reader = sitk.ImageSeriesReader()
    ct_series_names = ct_reader.GetGDCMSeriesFileNames(ct_folder)
    ct_reader.SetFileNames(ct_series_names)
    ct = ct_reader.Execute()
    sitk.WriteImage(ct, os.path.join(save_dir_ct, f"{patient_id}_0000.nii.gz"), imageIO='NiftiImageIO')
    print('Saved nifti CT')

    # Convert PET series
    pet_reader = sitk.ImageSeriesReader()
    pet_series_names = pet_reader.GetGDCMSeriesFileNames(pet_folder)
    pet_reader.SetFileNames(pet_series_names)
    pet = pet_reader.Execute()

    pet_img = dcmread(pet_series_names[0])  # read one of the images for header info
    suv_factor = bqml_to_suv(pet_img)
    pet = sitk.Multiply(pet, suv_factor)
    sitk.WriteImage(pet, os.path.join(save_dir_pt, f"{patient_id}_0001.nii.gz"), imageIO='NiftiImageIO')
    print('Saved nifti PET')

# Execution
for struct in structs:
    patient_id, struct_folder, ref_folder, convert = struct
    if convert=='N':
        continue

    # print('Converting RTStruct for patient {}'.format(num))
    # Get all the paths in order
    struct_file = os.listdir(struct_folder)[0]
    struct_path = os.path.join(struct_folder, struct_file)

    # Create the mask
    rtstruct = RTStructBuilder.create_from(dicom_series_path= ref_folder, rt_struct_path=struct_path)
    final_mask = load_merge_masks(rtstruct)

    # Load original DICOM image for reference
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(ref_folder)
    reader.SetFileNames(dicom_names)
    ref_img = reader.Execute()

    # Properly reference and convert the mask to an image object
    mask_img = sitk.GetImageFromArray(final_mask)
    mask_img.CopyInformation(ref_img)
    sitk.WriteImage(mask_img, os.path.join(save_dir_gt, f"{patient_id}.nii.gz"), imageIO="NiftiImageIO")

    print('Patient {} mask saved'.format(patient_id))
