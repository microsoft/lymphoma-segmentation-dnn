# Getting started

Welcome to our GitHub codebase for lymphoma lesion segmentation from PET/CT images. 

## Cloning the repository
To get started, the first step is the clone this repository to your local machine and navigate inside the resulting git directory:

```
git clone 'https://github.com/ahxmeds/lymphoma.segmentation.dnn.git'
cd lymphoma.segmentation
```

## Installing packages from `environment.yml` file
This code base was developed primarily using python=3.8.10, PyTorch=1.11.0, monai=1.2.0, with CUDA 11.3 on an Ubuntu 20.04 virtual machine, so the codebase has been tested only with these configurations. We hope that it will run in other suitable combinations of different versions of python, PyTorch, monai, and CUDA, but we cannot guarantee that. Proceed with caution!  

Firstly, we will use the [environment.yml](/environment.yml) file to create a conda environment (`lymphoma_seg`) and install all required packages as mentioned in the [environment.yml](/environment.yml) file. For this step, run,

```
conda env create --file environment.yml
```

If the above step is completed successfully without errors, you will have a new conda environment called `lymphoma_seg`. To activate this environment, use

```
conda activate lymphoma_seg
```

The environment can be deactivated using

```
conda deactivate
```

With the conda environment set up, you have all the necessary tools to start a training or inference experiment, except the training/test dataset. The next step is get your dataset in the format which can be used by our codebase, as explained in [/documentation/dataset_format.md](/documentation/dataset_format.md).
