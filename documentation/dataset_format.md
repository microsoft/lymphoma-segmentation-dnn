# Dataset format
In this work, the dataset consist of three components: CT and PET images and the corresponding lesion segmentation mask, all in NIFTI file format. If your dataset is in DICOM format, you can convert them to NIFTI using the method described in [dicom_to_nifti_conversion.md](./dicom_to_nifti_conversion.md). After converting DICOM images to NIFTI format, you may have to resample you CT (and/or GT) images to PET geometry (if your CT or GT images are not in PET geometry). If this is the case, use the functions `resample_ct_to_pt_geometry()` and `resample_gt_to_pt_geometry()` in [data_conversion/resample_ct2pt.py](./../data_conversion/resample_ct2pt.py). 

## Training cases filenaming convention
We follow a similar filenaming convention as used by [nnUNet](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md). Each training case is associated with a unique identifier, which is a unique name for that case. This identifier is used by our code to connect images (PET/CT) with the correct segmentation mask. We suggest using the unique identifier as `{PatientID}_{StudyDate}`.

A training case consists of images and their corresponding segmentation masks. 

**Images**: Our networks utilize two channel 3D images, the first channel being the CT images and the second channel being the PET image.  Both CT and PET **MUST** have the same geometry (same size, spacing, origin, direction) and must be (approximately) coregistered (if applicable). To resample CT images to PET resolution, use the function `resample_ct_to_pt_geometry()` in [data_conversion/resample_ct2pt.py](./../data_conversion/resample_ct2pt.py). Within a training case, all image geometries (input channels, corresponding segmentation) must match. Between training cases, they can of course differ. 

**Segmentations** must share the same geometry as their corresponding images (same size, spacing, origin, direction). Segmentations are 
integer maps with each value representing a semantic class; the background is represented by 0. In our work, we used segmentation masks with two classes: 0 for background and 1 for lesions. All masks in the training set **MUST** have 0s and 1s; the current version of code cannot handle negative images (images with no lesions) without changing some of the preprocessing transforms (like `RandCropByPosNegLabeld`, etc.) applied to the images before giving them as inputs to the network.  

Given a unique identifier for a case, {PatientID}_{StudyDate}, the CT, PET and GT image filenames should be:  
CT image: `{PatientID}_{StudyDate}_0000.nii.gz`,  
PET image: `{PatientID}_{StudyDate}_0001.nii.gz`,   
GT image: `{PatientID}_{StudyDate}.nii.gz`,  

**Important:** The input channels must be consistent! Concretely, **all images need the same input channels in the same 
order and all input channels have to be present every time**. This is also true for inference!


## Dataset folder structure
Create a folder named `lymphoma.segmentation` in the location of your choice. The is the master folder that stored all your datasets, the trained models and training/validation logs, predictions or any other results based on predictions. Go to the file [config.py](./../config.py) and update the variable `LYMPHOMA_SEGMENTATION_FOLDER` as the absolute path to the folder `lymphoma.segmentation`. Within `lymphoma.segmentation`, create a folder named `data`, which should be the location of your training and test datasets. After these steps, your directory structure is expected to look like this:

    └───lymphoma.segmentation/data
        ├── imagesTr
        ├── imagesTs  # optional
        ├── labelsTr  
        └── labelsTs # optional

- `imagesTr` contains the images (CT and PET) belonging to the training cases. Each corresponding CT and PET images should be in the same geometry (same size, spacing, origin, direction) in this folder.  
- `imagesTs` (optional) contains the images that belong to the test cases. Each corresponding CT and PET images should be in the same geometry (same size, spacing, origin, direction) in this folder.  
- `labelsTr` contains the images with the ground truth segmentation maps for the training cases. These should be in the same geometry (same size, spacing, origin, direction) as their corresponding PET/CT images in `imagesTr`.
- `labelsTs` (optional) contains the images with the ground truth segmentation maps for the test cases. These should be in the same geometry (same size, spacing, origin, direction) as their corresponding PET/CT images in `imagesTs`.


After moving all the training and test images and masks in the respective folders, the directory structure should look like this:

    └───lymphoma.segmentation/data/
        ├── imagesTr
        │   ├── Patient0001_20110502_0000.nii.gz
        │   ├── Patient0001_20110502_0001.nii.gz
        │   ├── Patient0002_20150514_0000.nii.gz
        │   ├── Patient0002_20150514_0001.nii.gz
        │   ├── ...
        ├── imagesTs  # optional
        │   ├── Patient0003_20190402_0000.nii.gz
        │   ├── Patient0003_20190402_0001.nii.gz
        │   ├── Patient0004_20150514_0000.nii.gz
        │   ├── Patient0004_20150514_0001.nii.gz
        │   ├── ...
        ├── labelsTr 
        │   ├── Patient0001_20110502.nii.gz
        │   ├── Patient0002_20110502.nii.gz 
        │   ├── ...
        └── labelsTs # optional
            ├── Patient0003_20190402.nii.gz
            ├── Patient0004_20160204.nii.gz 
            ├── ...
