# How to calculate test metrics on the test set predicted masks?
Once you have trained some models (as described in [trainddp.md](./trainddp.md)) and used them to perform inference (as described in [inference.md](./inference.md)) to generate predicted masks on the test images, you can proceed with the computation of test metrics. We compute three segmentation metrics: `Dice similarity coefficient (DSC)`, `false positive volume (FPV) in ml`, `false negative volume (FNV) in ml`. We also compute detection metrics such as `true positive (TP)`, `false positive (FP)`, and `false negative (FN)` lesion detections via three different criterion labeled as `Criterion1`, `Criterion2`, and `Criterion3`. These metrics have been defined in [metrics/metrics.py](./../metrics/metrics.py). 

## Step 1: Activate the required conda environment (`lymphoma_seg`) and navigate to `segmentation` folder
First, activate the conda environment `lymphoma_seg` using (created as described in [conda_env.md](./conda_env.md)):  

```
conda activate lymphoma_seg
cd segmentation
```

## Step 2: Run the script to compute test metrics
After this, run the following script in your terminal,
```
python calculate_test_metrics.py --fold=0 --network-name='unet' --input-patch-size=192
```

Alternatively, modify the [segmentation/calculate_test_metrics.sh](./../segmentation/calculate_test_metrics.sh) for your use-case (which contains the same bash script as above) and run:

```
bash calculate_test_metrics.sh
```

The test metrics will be written to `LYMPHOMA_SEGMENTATION_FOLDER/results/test_metrics/fold{fold}/{network_name}/{experiment_code}/testmetrics.csv`, as described in [results_format.md](./results_format.md) file. The relevant directory structure may then look like:

    └───lymphoma.segmentation/
            ├── data
            └── results
                ├── logs
                ├── models
                ├── predictions
                └── test_metrics
                    ├── fold0
                    │   └── unet
                    │       └── unet_fold0_randcrop192
                    │           └── testmetrics.csv   
                    └── fold1
                        └── unet
                            └── unet_fold1_randcrop192
                                └── testmetrics.csv