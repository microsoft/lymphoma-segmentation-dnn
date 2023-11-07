'''
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
'''

import SimpleITK as sitk 
import numpy as np  
import cc3d

#%%
def get_3darray_from_niftipath(
    path: str,
) -> np.ndarray:
    """Get a numpy array of a Nifti image using the filepath

    Args:
        path (str): path of the Nifti file

    Returns:
        np.ndarray: 3D numpy array for the image
    """
    image = sitk.ReadImage(path)
    array = np.transpose(sitk.GetArrayFromImage(image), (2,1,0))
    return array

def calculate_patient_level_lesion_suvmean_suvmax(
    ptarray: np.ndarray, 
    maskarray: np.ndarray,
    marker: str = 'SUVmean'
) -> np.float64:
    """Function to return the lesion SUVmean or SUVmax for all lesions in 
    a 3D PET image using the corresponding 3D segmentation mask 

    Args:
        ptarray (np.ndarray): numpy ndarray for 3D PET image
        maskarray (np.ndarray): numpy ndarray for 3D mask image
        marker (str, optional): Whether you want to calculate SUVmean or SUVmax . 
        Defaults to 'SUVmean'.

    Returns:
        np.float64: patient-level SUVmean or SUVmax
    """
    prod = np.multiply(ptarray, maskarray)
    num_nonzero_voxels = len(np.nonzero(maskarray)[0])

    if num_nonzero_voxels == 0:
        return 0.0
    else:
        if marker == 'SUVmean':
            return np.sum(prod)/num_nonzero_voxels
        elif marker == 'SUVmax':
            return np.max(prod)

#%%
def calculate_patient_level_tmtv(
    maskarray: np.ndarray,
    spacing: tuple
) -> np.float64:
    """Function to return the total metabolic tumor volume (TMTV) in cm^3 using 
    3D mask containing 0s for background and 1s for lesions/tumors
    Args:
        maskarray (np.ndarray): numpy ndarray for 3D mask image

    Returns:
        np.float64: 
    """
    voxel_volume_cc = np.prod(spacing)/1000 # voxel volume in cm^3

    num_lesion_voxels = len(np.nonzero(maskarray)[0])
    tmtv_cc = voxel_volume_cc*num_lesion_voxels
    return tmtv_cc

#%%

def calculate_patient_level_lesion_count(
    maskarray: np.ndarray,
) -> int:
    """Function to return the total number of lesions using the 3D segmentation mask 
    Args:
        maskarray (np.ndarray): numpy ndarray for 3D mask image

    Returns:
        int: _description_
    """
    _, num_lesions = cc3d.connected_components(maskarray, connectivity=18, return_N=True)
    return num_lesions

#%%
def calculate_patient_level_tlg(
    ptarray: np.ndarray,
    maskarray: np.ndarray,
    spacing: tuple
) -> np.float64:
    """Function to return the total lesion glycolysis (TLG) using a 3D PET image 
    and the corresponding 3D segmentation mask (containing 0s for background and
    1s for lesion/tumor)
    TLG = SUV1*V1 + SUV2*V2 + ... + SUVn*Vn, where SUV1...SUVn are the SUVmean 
    values of lesions 1...n with volumes V1...Vn, respectively

    Args:
        ptarray (np.ndarray): numpy ndarray for 3D PET image
        maskarray (np.ndarray): numpy ndarray for 3D mask image

    Returns:
        np.float64: total lesion glycolysis in cm^3 (assuming SUV is unitless)
    """
    voxel_volume_cc = np.prod(spacing)/1000 # voxel volume in cm^3

    labels_out, num_lesions = cc3d.connected_components(maskarray, connectivity=18, return_N=True)
    if num_lesions == 0:
        return 0.0
    else:
        _, lesion_num_voxels = np.unique(labels_out, return_counts=True)
        lesion_num_voxels = lesion_num_voxels[1:]
        lesion_mtvs = voxel_volume_cc*lesion_num_voxels
        lesion_suvmeans = []
        
        for i in range(1, num_lesions+1):
            mask = np.zeros_like(labels_out)
            mask[labels_out == i] = 1
            prod = np.multiply(mask, ptarray)
            num_nonzero_voxels = len(np.nonzero(mask)[0])
            lesion_suvmeans.append(np.sum(prod)/num_nonzero_voxels)
        
        tlg = np.sum(np.multiply(lesion_mtvs, lesion_suvmeans))
        return tlg
#%%
def calculate_patient_level_dissemination(
    maskarray: np.ndarray,
    spacing: tuple
) -> np.float64:
    """Function to return the tumor dissemination (Dmax) using 3D segmentation mask
    Dmax = max possible distance between any two foreground voxels in a patient;
    these two voxels can come form the same lesions (in case of one lesion) 
    or from different lesions (in case of multiple lesions) 
   
    Args:
        maskarray (np.ndarray): numpy array for 3D mask image

    Returns:
        np.float64: dissemination value in cm
    """
    maskarray = maskarray.astype(np.int8)
    nonzero_voxels = np.argwhere(maskarray == 1)
    distances = np.sqrt(np.sum(((nonzero_voxels[:, None] - nonzero_voxels) * spacing)**2, axis=2))
    farthest_indices = np.unravel_index(np.argmax(distances), distances.shape)
    dmax = distances[farthest_indices]/10  # converting to cm
    del maskarray 
    del nonzero_voxels
    del distances 
    return dmax 

#%%
def calculate_patient_level_dice_score(
    gtarray: np.ndarray,
    predarray: np.ndarray, 
) -> np.float64:
    """Function to return the Dice similarity coefficient (Dice score) between
    2 segmentation masks (containing 0s for background and 1s for lesions/tumors)

    Args:
        maskarray_1 (np.ndarray): numpy ndarray for the first mask
        maskarray_2 (np.ndarray): numpy ndarray for the second mask

    Returns:
        np.float64: Dice score
    """
    dice_score = 2.0*np.sum(predarray[gtarray == 1])/(np.sum(gtarray) + np.sum(predarray))
    return dice_score
#%%
def calculate_patient_level_iou(
    gtarray: np.ndarray,
    predarray: np.ndarray, 
) -> np.float64:
    """Function to return the Intersection-over-Union (IoU) between
    2 segmentation masks (containing 0s for background and 1s for lesions/tumors)

    Args:
        maskarray_1 (np.ndarray): numpy ndarray for the first mask
        maskarray_2 (np.ndarray): numpy ndarray for the second mask

    Returns:
        np.float64: Dice score
    """
    intersection = np.sum(predarray[gtarray == 1])
    union = np.sum(gtarray) + np.sum(predarray) - intersection
    iou = intersection/union
    return iou

def calculate_patient_level_intersection(
    gtarray: np.ndarray,
    predarray: np.ndarray, 
) -> np.float64:
    """Function to return the Intersection etween
    2 segmentation masks (containing 0s for background and 1s for lesions/tumors)

    Args:
        maskarray_1 (np.ndarray): numpy ndarray for the first mask
        maskarray_2 (np.ndarray): numpy ndarray for the second mask

    Returns:
        np.float64: Dice score
    """
    intersection = np.sum(predarray[gtarray == 1])
    return intersection
#%%

def calculate_patient_level_false_positive_volume(
    gtarray: np.ndarray,
    predarray: np.ndarray,
    spacing: tuple
) -> np.float64:
    # compute number of voxels of false positive connected components in prediction mask
    pred_connected_components = cc3d.connected_components(predarray, connectivity=18)
    
    false_positive = 0
    for idx in range(1,pred_connected_components.max()+1):
        comp_mask = np.isin(pred_connected_components, idx)
        if (comp_mask*gtarray).sum() == 0:
            false_positive += comp_mask.sum()
    
    voxel_volume_cc = np.prod(spacing)/1000
    return false_positive*voxel_volume_cc

#%%
def calculate_patient_level_false_negative_volume(
    gtarray: np.ndarray,
    predarray: np.ndarray,
    spacing: tuple
) -> np.float64:
    # compute number of voxels of false negative connected components (of the ground truth mask) in the prediction mask
    gt_connected_components = cc3d.connected_components(gtarray, connectivity=18)
    
    false_negative = 0
    for idx in range(1,gt_connected_components.max()+1):
        comp_mask = np.isin(gt_connected_components, idx)
        if (comp_mask*predarray).sum() == 0:
            false_negative += comp_mask.sum()

    voxel_volume_cc = np.prod(spacing)/1000
    return false_negative*voxel_volume_cc

# %%
def is_suvmax_detected(
    gtarray: np.ndarray,
    predarray: np.ndarray,
    ptarray: np.ndarray,
) -> bool:
    prod = np.multiply(gtarray, ptarray)
    max_index = np.unravel_index(np.argmax(prod), prod.shape)
    if predarray[max_index] == 1:
        return True
    else:
        return False


def calculate_patient_level_tp_fp_fn(
    gtarray: np.ndarray,
    predarray: np.ndarray,
    criterion: str,
    threshold: np.float64 = None,
    ptarray: np.ndarray = None,
) -> (int, int, int):
    """Calculate patient-level TP, FP, and FN (for detection based metrics)
    via 3 criteria:

    criterion1: A predicted lesion is TP if any one of it's foreground voxels 
    overlaps with GT foreground. A predicted lesions that doesn't overlap with any 
    GT foreground is FP. As soon as a lesion is predicted as TP, it is removed
    from the set of GT lesions. The lesions that remain in the end in the GT lesions
    are FN. `criterion1` is the weakest detection criterion.

    criterion2: A predicted lesion is TP if more than `threshold`% of it's volume 
    overlaps with foreground GT. A predicted lesion is FP if it overlap fraction
    with foreground GT is between 0% and `threshold`%. As soon as a lesion is 
    predicted as TP, it is removed from the set of GT lesions. The lesions that 
    remain in the end in the GT lesions are FN. `criterion2` can be hard or weak 
    criterion based on the value of `threshold`.

    criterion3: A predicted lesion is TP if it overlaps with one the the GT lesion's 
    SUVmax voxel, hence this criterion requires the use of PET data (`ptarray`). A 
    predicted lesion that doesn't overlap with any GT lesion's SUVmax voxel is 
    considered FP. As soon as a lesion is predicted as TP, it is removed from the 
    set of GT lesions. The lesions that remain in the end in the GT lesions are FN. 
    `criterion3` is likely an easy criterion since a network is more likely to segment 
    high(er)-uptake regions`.

    Args:
        int (_type_): _description_
        int (_type_): _description_
        gtarray (_type_, optional): _description_. Defaults to None, ptarray: np.ndarray = None, )->(int.
    """
    
    gtarray_labeled_mask, num_lesions_gt = cc3d.connected_components(gtarray, connectivity=18, return_N=True)
    predarray_labeled_mask, num_lesions_pred = cc3d.connected_components(predarray, connectivity=18, return_N=True)
    gt_lesions_list = list(np.arange(1, num_lesions_gt+1))
    #initial values for TP, FP, FN
    TP = 0
    FP = 0 
    FN = num_lesions_gt 

    if criterion == 'criterion1':
        FN = 0 # for this criterion we are counting the number of FPs from 0 onwards, hence the reassignment
        for i in range(1, num_lesions_pred+1):
            pred_lesion_mask = np.where(predarray_labeled_mask == i, 1, 0)
            if np.any(pred_lesion_mask & (gtarray_labeled_mask > 0)):
                TP += 1
            else:
                FP += 1
        for j in range(1, num_lesions_gt+1):
            gt_lesion_mask = np.where(gtarray_labeled_mask == j, 1, 0)
            if not np.any(gt_lesion_mask & (predarray_labeled_mask > 0)):
                FN += 1

    elif criterion == 'criterion2':
        for i in range(1, num_lesions_pred+1):
            max_iou = 0
            match_gt_lesion = None 
            pred_lesion_mask = np.where(predarray_labeled_mask == i, 1, 0)
            for j in range(1, num_lesions_gt+1):
                gt_lesion_mask = np.where(gtarray_labeled_mask == j, 1, 0)
                iou = calculate_patient_level_iou(gt_lesion_mask, pred_lesion_mask)
                if iou > max_iou:
                    max_iou = iou
                    match_gt_lesion = j
            if max_iou >= threshold:
                TP += 1
                gt_lesions_list.remove(match_gt_lesion)
            else:
                FP += 1
        FN = len(gt_lesions_list)

    elif criterion == 'criterion3':
        for i in range(1, num_lesions_pred+1):
            max_iou = 0
            match_gt_lesion = None
            pred_lesion_mask = np.where(predarray_labeled_mask == i, 1, 0)
            for j in range(1, num_lesions_gt+1):
                gt_lesion_mask = np.where(gtarray_labeled_mask == j, 1, 0)
                iou = calculate_patient_level_iou(gt_lesion_mask, pred_lesion_mask)
                if iou > max_iou:
                    max_iou = iou 
                    match_gt_lesion = j
            
            # match_gt_lesion has been defined with has the maximum iou with pred lesion i
            arr_gt_lesion = np.where(gtarray_labeled_mask == match_gt_lesion, 1, 0)
            if is_suvmax_detected(arr_gt_lesion, pred_lesion_mask, ptarray):
                TP += 1
                gt_lesions_list.remove(match_gt_lesion)
            else:
                FP += 1
        
        FN = len(gt_lesions_list)

    else:
        print('Invalid criterion. Choose between criterion1, criterion2, or criterion3')
        return 
    
    return TP, FP, FN

