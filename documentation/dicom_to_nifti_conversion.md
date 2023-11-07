# Converting DICOM series to 3D NIFTI files

PET/CT images are usually stored in DICOM format (the format from hell). In our work, we have converted DICOM PET/CT and RTSTRUCT images as NIFIT images for use by our networks. Unlike the DICOM series that consists of several (axial) `.dcm` images within a folder for one case, the NIFTI images are just one file (`.nii.gz`) which stores the entire 3D array + associated metadata. Hence, NIFTI images are much easier to handle and suitable format to use in deep learning applications. 

Here, we provide the script [dicom_to_nifti.py](./../data_conversion/dicom_to_nifti.py) for converting DICOM series (for PET and CT) and DICOM RTSTRUCT (for segmentation masks in DICOM format) to 3D NIFTI files. Before using this code, you need to create two specific files: `dicom_ctpt_to_nifti_conversion_file.csv` and `dicom_rtstruct_to_nifti_conversion_file.csv`. Examples of these files are given in [here](./../data_conversion/dicom_ctpt_to_nifti_conversion_file.csv) and [here](./../data_conversion/dicom_rtstruct_to_nifti_conversion_file.csv), respectively. Both these files are used by [dicom_to_nifti.py](./../data_conversion/dicom_to_nifti.py) for performing the required conversions. **DO NOT FORGET TO READ THE `VERY IMPORTANT NOTES` SECTION AT THE BOTTOM OF THIS DOCUMENT**.

## Creating the `dicom_ctpt_to_nifti_conversion_file.csv` file
`dicom_ctpt_to_nifti_conversion_file.csv` must be a .csv file and its contents must look like this:

| PatientID | CT_dir | PET_dir | convert | 
| ----------|--------|---------|---------|
| Patient00001_28071996 | path/to/ct/dicom/series/directory/for/Patient00001_28071996 | path/to/pet/dicom/series/directory/for/Patient00001_28071996 | Y |
| Patient00002_02021996 | path/to/ct/dicom/series/directory/for/Patient00002_02021996 | path/to/pet/dicom/series/directory/for/Patient00002_02021996 | Y |

Here, the first column is `PatientID`. For the purpose of illustration, we are using the unique identifier `{PatientID}_{StudyDate}`, as described in [dataset_format.md](./dataset_format.md), but you can use any other naming convention too. The second and third columns should be the path to the DICOM directories for CT and PET respectively for patient with ID `PatientID`. The last column should be either `Y` or `N`, for whether to convert (to NIFTI) or not. Rows with `convert=N` are ignored in [dicom_to_nifti.py](./../data_conversion/dicom_to_nifti.py) during conversion. Populate this .csv file with this information corresponding to your custom DICOM data. 


## Creating the `dicom_rstruct_to_nifti_conversion_file.csv` file
`dicom_rtstruct_to_nifti_conversion_file.csv` must be a .csv file and its contents must look like this:

| PatientID | RTSTRUCT_dir | REF_dir | convert | 
| ----------|--------------|---------|---------|
| Patient00001_28071996 | path/to/dicom/rtstruct/directory/for/Patient00001_28071996 | path/to/reference/dicom/series/for/Patient00001_28071996 | Y |
| Patient00002_02021996 | path/to/dicom/rtstruct/directory/for/Patient00002_02021996 | path/to/reference/dicom/series/for/Patient00002_02021996 | Y |

Here, the first column in the `PatientID`. The second column is the path to RTSTRUCT directory for patient with ID `PatientID`. The third column is the path to the directory that stores the reference image on which the RTSTRUCT was created. This reference image could be either PET or CT depending on which image was used to create RTSTRUCT annotations. BE CAREFUL with assigning the correct reference image, otherwise the code [dicom_to_nifti.py](./../data_conversion/dicom_to_nifti.py) will fail. The last column is the same as the previous step.


## Updating the `save_dir_ct`, `save_dir_pt`, and `save_dir_gt` in `dicom_to_nifti.py`
Go to the middle of [dicom_to_nifti.py](./../data_conversion/dicom_to_nifti.py) file (around line number 87-95) and update the values of variables `save_dir_ct`, `save_dir_pt`, and `save_dir_gt` with the path locations to directories (in your local machine) where you want the converted images in NIFTI format to be written, corresponding to CT, PET, and GT (ground truth) masks, respectively. 
```
############################################################################################
########  Update the three variables below with the locations of your choice  ##############
############################################################################################
save_dir_ct = '' # path to directory where your new CT files in NIFTI format will be written
save_dir_pt = '' # path to directory where your new PET files in NIFTI format will be written
save_dir_gt = '' # path to directory where your new GT files in NIFTI format will be written
############################################################################################
############################################################################################
############################################################################################
```

## Running conversion script  `dicom_to_nifti.py`
This step assumes that you have already cloned this repository, create a conda environment `lymphoma_seg` with all the necessary packages installed from [environment.yml](./../environment.yml) file. If you haven't done these steps, first finish them using [conda_env.md](./conda_env.md) before proceeding further. Also, read the next section `VERY IMPORTANT NOTES` before running the conversion script below (as you might have to update `dicom_to_nifti.py` further), 
```
conda activate lymphoma_seg
cd data_conversion
python dicom_to_nifti.py
``` 


## VERY IMPORTANT NOTES
-  [dicom_to_nifti.py](./../data_conversion/dicom_to_nifti.py) uses [rt-utils](https://github.com/qurit/rt-utils) for converting DICOM RTSTRUCT to 3D numpy arrays which are eventually saved as 3D NIFTI masks. The code contains a function `get_filtered_roi_list(.)`, as given below:
    ```
    def get_filtered_roi_list(rois):
        filtered_rois = []
        for roi in rois:
            if roi.endswith('PETEdge'):
                filtered_rois.append(roi)
            else:
                pass
        return filtered_rois
    ```

    The `rois` which is passed as an argument to this function is list of ROIs within the RTSTRUCT (as extracted by the `rt_utils.RTStructBuilder`). In our datasets, all the ROIs in the RTSTRUCT files ending with the string `PETEdge` corresponded to lesions, hence we use `get_filtered_roi_list(.)` to filter only the ROIs for lesions. Your dataset may or may not be like this, hence **BE VERY CAUTIOUS WHILE USING THIS CODE!!!!!!!!  Update the code accordingly depending on your use-case**. 

- This code also assumes that the original DICOM PET series intensities were in units of Bq/ml. This code converts performs decay-correction of PET intensities and converts them to SUV values, before converting them to NIFTI images. The CT images intensities remains the same (i.e., Hounsfield Units (HU)) before and after conversion to NIFTI. 

- After converting to NIFTI format, your CT, PET, and GT still might not be in the same geometry. For example, your high-resolution CT images could have a matrix size much larger than your lower-resolution PET images. You must resample (and resave) the CT images (and also GT masks, if application) to the geometry of PET images. The final PET, CT and GT mask for a specific `PatientID` should all have the same size, spacing, origin, and direction. To perform this resampling, use the functions `resample_ct_to_pt_geometry()` and/or `resample_gt_to_pt_geometry()` in [resample_ct2pt.py](./../data_conversion/resample_ct2pt.py). If you do not perform this final resampling of PET/CT/GT images to the same geometry, the subsequent training code will fail, as described in [dataset_format.md](./dataset_format.md). 



