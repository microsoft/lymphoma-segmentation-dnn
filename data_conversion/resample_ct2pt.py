'''
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
'''

import numpy as np
import os
import SimpleITK as sitk
import pandas as pd
from pathlib import Path
import glob


def resample_ct_to_pt_geometry(
    ctpath: str, 
    ptpath: str,
    savedir: str = ''
):
    """ Function to resample CT images to the corresponding PET image geometry.
    This functions assumes that the CT and PET are already coregistered.

    Args:
        ctpath (str): path to NIFTI file for (high-resolution) CT image 
        ptpath (str): path to NIFTI file for PET image
        savedir (str, optional): Directory to write the downsampled CT NIFTI image. Defaults to ''.
    """
    ctimg = sitk.ReadImage(ctpath)
    ptimg = sitk.ReadImage(ptpath)
    resampled_ctimg = sitk.Resample(ctimg, ptimg, interpolator=sitk.sitkLinear, defaultPixelValue=-1024)
    resampled_ct_filepath = os.path.join(savedir, os.path.basename(ctpath))
    
    sitk.WriteImage(resampled_ctimg, resampled_ct_filepath)
    print('Resampled CT to PET geometry')
    print(f'Saving the low-resolution CT NIFTI image at {resampled_ct_filepath}')
 
def resample_gt_to_pt_geometry(
    gtpath: str, 
    ptpath: str,
    savedir: str = ''
):
    """ Function to resample GT images (if applicable) to the corresponding PET image geometry.
    You may or may not need to do this resampling. Do this if your ground truth segmentations 
    were performed on CT images, and hence your GT masks are in the geometry of CT instead of PET.
    If the annoatations were performed on PET, then the GT mask and PET should (ideally) be in the 
    same geometry and hence this step may not be required.

    Args:
        gtpath (str): path to NIFTI file for (high-resolution) GT image 
        ptpath (str): path to NIFTI file for PET image
        savedir (str, optional): Directory to write the downsampled GT NIFTI image. Defaults to ''.
    """
    gtimg = sitk.ReadImage(gtpath)
    ptimg = sitk.ReadImage(ptpath)
    resampled_gtimg = sitk.Resample(gtimg, ptimg, interpolator=sitk.sitkNearestNeighbor, defaultPixelValue=0)
    resampled_gt_filepath = os.path.join(savedir, os.path.basename(gtpath))
    
    sitk.WriteImage(resampled_gtimg, resampled_gt_filepath)
    print('Resampled GT to PET geometry')
    print(f'Saving the low-resolution CT NIFTI image at {resampled_gt_filepath}')