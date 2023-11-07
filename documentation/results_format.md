# Results folder format
In this work, results include: trained models, training and validation logs, predicted masks, metrics on the test set, etc. These will all be written to a folder called `results` as defined in the variable `RESULTS_FOLDER` in the `config.py` (`./../config.py`) file. This folder will be next to the `data` folder, as explained in [dataset_format.md](LINK).

## Results folder/filenaming convention

### `logs` and `models` folders
While a model is training (see `trainddp.md` for details), the following two folders will be created within `results` folder: `logs` and `models` and the directory structure may look like this:

    └───lymphoma.segmentation/
        ├── data
        └── results
            ├── logs
            │    ├── fold0
            │    │   └── unet
            │    │       └── unet_fold0_rancrop192
            │    │           ├── trainlog_gpu0.csv
            │    │           ├── trainlog_gpu1.csv
            │    │           ├── validlog_gpu0.csv
            │    │           └── validlog_gpu1.csv
            │    └── fold1
            │        └── unet
            │            └── unet_fold1_rancrop192
            │                ├── trainlog_gpu0.csv
            │                ├── trainlog_gpu1.csv
            │                ├── validlog_gpu0.csv
            │                └── validlog_gpu1.csv
            ├── models
            │    ├── fold0
            │    │   └── unet
            │    │       └── unet_fold0_rancrop192
            │    │           ├── model_ep=0002.csv
            │    │           ├── model_ep=0004.csv
            │    │           ├── model_ep=0006.csv
            │    │           ├── model_ep=0008.csv
            │    │           ├── ...
            │    └── fold1
            │        └── unet
            │            └── unet_fold1_rancrop192
            │                ├── model_ep=0002.csv
            │                ├── model_ep=0004.csv
            │                ├── model_ep=0006.csv
            │                ├── model_ep=0008.csv
            │                ├── ...
            ├── ...  


This directory stucture shows that so far, the model `unet` has been (or is being) trained on two folds: `fold0` and `fold1`. Within the `logs` or `models` folder, the directory structure is `{logs_or_models}/fold{fold}/{network_name}/{experiment_code}`, where the `experiment_code` is defined as `{network_name}_fold{fold}_randcrop{input_patch_size}`. The above directory structure shows that for both folds `fold0` and `fold1`, the `experiment_code` is `{unet}_fold{0 or 1}_randcrop{192}`, meaning we trained/are training `unet` for fold 0 or 1 with an `input_patch_size = 192`. If you train other networks (like `segresnet`, `dynunet`, or `swinunetr` as was the case in this work), they will appear accordingly within the framework of the above directory structure.

Since the training in this work was carried out using the PyTorch's `torch.nn.parallel.DistributedDataParallel`, the `trainlog_gpu0.csv`, `trainlog_gpu1.csv`, `validlog_gpu0.csv`, `validlog_gpu1.csv` store the training and validation logs on accumulated on GPU with deviceids 0 and 1. All the `validlog_gpu[i].csv` are identical and hence redundant so you can use any one of them analysis (we will resolve this to save only one file, in the later versions). All the `trainlog_gpu[i].csv` are NOT identical, hence each file separately stores the loss accumulated using the distributed data on two GPUs. In our work, we used 4 GPUs, but the above directory structure only shows training on 2 GPUs for the purpose of illustration. The typical `trainlog_gpu[i].csv` file looks like this:

```
Loss
0.6536665889951918
0.6449973914358351
0.6385666595564948
0.6357755064964294
...
```

where each line shows the mean `DiceLoss` on the training inputs (averaged over all batches) at epoch `j+1` with `j` in the range `np.arange(0, epochs)`; `epochs` is the total number of epochs for which we are running the training. Similarly, a typical `validlog_gpu[i].csv` file looks like this:

```
Metric
0.0011193332029506564
0.001015653251670301
...
```
where each line shows the mean `DiceMetric` on the validation inputs at epoch `j` with `j` in the range `np.arange(2, epochs+1, val_interval)`, `epochs` is the total number of epochs for which we are running the training and `val_interval` (default=2) is the epoch interval at which we are running validation, computing Dice metric and saving the trained model. The variables `val_internal`, `epochs`, etc. can be set in `train.sh` script which is used for running the training.  

The saved models are saved in the similar way under the correspding /fold/network/experiment_code folder with filenames `model_ep=0002.pth`, `model_ep=0004.pth`, etc. In this case, `val_interval = 2` (for example), so the models are saved at interval of 2 starting from the second epoch.


### `predictions` and `test_metrics` folders
After the trained models are used for predicting the segmentation masks on test images (see `inference.md` for details), based on the `fold`, `network_name` and `experiment_code`, the predicted masks will be written to `LYMPHOMA_SEGMENTATION_FOLDER/results/predictions/fold{fold}/{network_name}/{experiment_code}`. Once the predicted masks have been generated and saved, the metrics computed on the test set using the test ground truth and predicted masks will be written to `LYMPHOMA_SEGMENTATION_FOLDER/results/test_metrics/fold{fold}/{network_name}/{experiment_code}/testmetrics.csv`. We compute three segmentation metrics: `Dice similarity coefficient (DSC)`, `false positive volume (FPV) in ml`, `false negative volume (FNV) in ml`. We also compute detection metrics such as `true positive (TP)`, `false positive (FP)`, and `false negative (FN)` lesion detections via three different criterion labeled as `Criterion1`, `Criterion2`, and `Criterion3`. These metrics have been defined in [metrics/metrics.py](./../metrics/metrics.py). After running inference and calculating the test metrics, the (relevant) directory structure may look like:

    └───lymphoma.segmentation/
            ├── data
            └── results
                ├── logs
                ├── models
                ├── predictions
                │   ├── fold0
                │   │   └── unet
                │   │       └── unet_fold0_randcrop192
                │   │           ├── Patient0003_20190402.nii.gz
                │   │           ├── Patient0004_20160204.nii.gz 
                │   │           ├── ...
                │   └── fold1
                │       └── unet
                │           └── unet_fold1_randcrop192
                │               ├── Patient0003_20190402.nii.gz
                │               ├── Patient0004_20160204.nii.gz 
                │               ├── ...
                │
                └── test_metrics
                    ├── fold0
                    │   └── unet
                    │       └── unet_fold0_randcrop192
                    │           └── testmetrics.csv   
                    └── fold1
                        └── unet
                            └── unet_fold1_randcrop192
                                └── testmetrics.csv

The predicted masks are in the same geometry (same size, spacing, origin, direction) as their corresponding ground truth masks. A typical `testmetrics.csv` file looks like:

| PatientID | DSC | FPV | FNV | TP_C1 | FP_C1 | FN_C1 | TP_C2 | FP_C2 | FN_C2 | TP_C3 | FP_C3 | FN_C3 |
|-----------|-----|-----|-----|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Patient0003_20190402 | 0.7221043699618158 | 17.5164623503173 | 1.173559512304143 | 3 | 6 | 2 | 2 | 7  | 3 | 3 | 6 | 2 | 
| Patient0004_20160204 | 0.0807955251709131 | 53.4186903933997 | 5.563541391664086 | 2 | 8 | 1 | 0 | 10 | 3 | 2 | 8 | 1 |

Here, all the metrics are at the patient level and FPV and FNV are expressed in ml.

### `test_lesion_measures` folder
In this work, we have performed further analyses on the predicted segmentation masks on the test set and compared them to the ground truth masks. These include comparing the patient-level lesion SUV<sub>mean</sub>, lesion SUV<sub>max</sub>, number of lesions, total metabolic tumor volume (TMTV) in ml, total lesion glycolysis (TLG) in ml, lesion dissemination (D<sub>max</sub>) in cm. These metrics have been defined in [metrics/metrics.py](./../metrics/metrics.py). The test set predicted lesion measures are written to `LYMPHOMA_SEGMENTATION_FOLDER/results/test_lesion_measures/fold{fold}/{network_name}/{experiment_code}/testlesionmeasures.csv`. After generating `testlesionmeasures.csv` files, the relevant directory structure may look like:

    └───lymphoma.segmentation/
            ├── data
            └── results
                ├── logs
                ├── models
                ├── predictions
                ├── test_metrics
                └── test_lesion_measures
                    ├── fold0
                    │   └── unet
                    │       └── unet_fold0_randcrop192
                    │           └── testlesionmeasures.csv   
                    └── fold1
                        └── unet
                            └── unet_fold1_randcrop192
                                └── testlesionmeasures.csv

A typical `testlesionmeasures.csv` file looks like:

| PatientID | DSC | SUVmean_orig | SUVmean_pred | SUVmax_orig | SUVmax_pred | LesionCount_orig | LesionCount_pred | TMTV_orig | TMTV_pred | TLG_orig | TLG_pred | Dmax_orig | Dmax_pred |
|-----------|-----|--------------|--------------|-------------|-------------|------------------|------------------|-----------|-----------|----------|----------|----------|-----------|
| Patient0003_20190402 | 0.7221043699618158  | 2.935304139385291 | 4.362726242681123 | 6.1822732035904515 | 7.827266273892102 | 3 | 4 | 13.691527643548337 | 18.6272625128359097 | 40.18879776661558 | 50.2728492927217289 | 15.837606584884108 | 25.82763813918739 | 
| Patient0004_20160204 | 0.0807955251709131  | 8.72882540822585 | 12.71524350987 | 40.294842200490244 | 45.9483628492382 | 9 | 6 | 20.732884717373196 | 16.756373846353748 | 180.9737309068245 | 120.2387139879348 | 14.737477375372881 | 7.652628627281008 |

Here, all the lesion measures are at the patient level. TMTV and TLG are expressed in ml and D<sub>max</sub> in cm.
