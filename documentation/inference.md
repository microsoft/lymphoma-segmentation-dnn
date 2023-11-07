# How to run inference on test images using your trained model?

Once your have trained some models using the training script described in [trainddp.md](./trainddp.md), you have model(s) that could be used for predicting the segmentation masks for test images. Running inference primarily uses three files from this codebase: [config.py](./../config.py), [segmentation/initialize_train.py](./../segmentation/initialize_train.py), and [segmentation/inference.py](./../segmentation/inference.py). Ensure that the [config.py](./../config.py) is correctly initialized (as described in [trainddp.md](./trainddp.md)) so that the inference code can find the path to the test images.   

## Step 1: Activate the required conda environment (`lymphoma_seg`) and navigate to `segmentation` folder
First, activate the conda environment `lymphoma_seg` using (created as described in [conda_env.md](./conda_env.md)):  

```
conda activate lymphoma_seg
cd segmentation
```

## Step 2: Run the inference script
After this, run the following script in your terminal. Note: we run the inference only on one GPU (denoted by `cuda:0` in your machine).  
```
python inference.py --fold=0 --network-name='unet' --input-patch-size=192 --num_workers=2  --sw-bs=2 
``` 

- `inference.py` is the inference code that this script is using.

- `--fold` defines which fold's trained model you want to use for inference. When training script is run for the first time, two files, namely, `train_filepaths.csv` and `test_filepaths.csv` gets created within the folder `WORKING_FOLDER/data_split`, where the former contains the filepaths (CT, PT, mask) for training images (from `imagesTr` and `labelsTr` folders as described in `dataset_format.md`), and the latter contains the filepaths for test images (from `imagesTs` and `labelsTs`), respectively. The purpose of setting `fold` in this case is not to point to the specific fold dataset (since we are only using the test set for inference), but to define which fold's trained model to use. Defaults to 0.

- `--network-name` defines the name of the network. In this work, we have trained UNet, SegResNet, DynUNet and SwinUNETR (adpated from MONAI [LINK]). Hence, the `--network-name` should be set to one of `unet`, `segresnet`, `dynunet`, or `swinunetr`. Defaults to `unet`.

- `--input-patch-size` defines the size of the cubic input patch that is cropped from the input images during training. We used `input-patch-size` of 224 for UNet, 192 for SegResNet, 160 for DynUNet and 128 for SwinUNETR. Defaults to 192.

- `--num-workers` defines the `num_workers` argument inside training and validation DataLoaders. Defaults to 2.

- `--sw-bs` defines the batch size for performing the sliding-window inference via `monai.inferers.sliding_window_inference` on the test inputs. Defaults to 2. 


Alternatively, modify the [segmentation/predict.sh](./../segmentation/predict.sh) script for your use-case (which contains the same bash script as above) and run:

```
bash predict.sh
```

The predicted masks will be written to `LYMPHOMA_SEGMENTATION_FOLDER/results/predictions/fold{fold}/{network_name}/{experiment_code}`, as described in [results_format.md](./results_format.md) file. The predicted masks are assigned the same filenames as the corresponding original ground truth segmentation masks. The relevant directory structure may then look like:

    └───lymphoma.segmentation/
            ├── data
            └── results
                ├── logs
                ├── models
                └── predictions
                    └── fold0
                        └── unet
                            └── unet_fold0_randcrop192
                                ├── Patient0003_20190402.nii.gz
                                ├── Patient0004_20160204.nii.gz 
                                ├── ...

