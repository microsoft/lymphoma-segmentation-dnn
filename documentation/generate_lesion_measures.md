# How to generate lesion measures from the test set predicted masks?
Once you have performed the inference and saved the network predicted masks in NIFTI format (as described in [inference.md](./inference.md)), you can proceed with the generation of lesion measures from test set predicted and ground truth lesions masks. We compute six different patient level lesion measures: patient-level lesion SUV<sub>mean</sub>, lesion SUV<sub>max</sub>, number of lesions, total metabolic tumor volume (TMTV) in ml, total lesion glycolysis (TLG) in ml, and lesion dissemination (D<sub>max</sub>) in cm. These metrics have been defined in [metrics/metrics.py](./../metrics/metrics.py) and have been shown to be prognostic biomarkers in lymphoma. 

## Step 1: Activate the required conda environment (`lymphoma_seg`) and navigate to `segmentation` folder
First, activate the conda environment `lymphoma_seg` using (created as described in [conda_env.md](./conda_env.md)):  

```
conda activate lymphoma_seg
cd segmentation
```

## Step 2: Run the script to compute test metrics
After this, run the following script in your terminal,
```
python generate_lesion_measures.py --fold=0 --network-name='unet' --input-patch-size=192
```

Alternatively, modify the [segmentation/generate_lesion_measures.sh](./../segmentation/generate_lesion_measures.sh) for your use-case (which contains the same bash script as above) and run:

```
bash generate_lesion_measures.sh
```

The ground truth and predicted lesion measures on the test set will be written to `LYMPHOMA_SEGMENTATION_FOLDER/results/test_lesion_measures/fold{fold}/{network_name}/{experiment_code}/testlesionmeasures.csv`, as described in [results_format.md](./results_format.md) file. The relevant directory structure may then look like:

    └───lymphoma.segmentation/
            ├── data
            └── results
                ├── logs
                ├── models
                ├── predictions
                └── test_metrics
                └── test_lesion_measures
                    ├── fold0
                    │   └── unet
                    │       └── unet_fold0_randcrop192
                    │           └── testlesionmeasures.csv   
                    └── fold1
                        └── unet
                            └── unet_fold1_randcrop192
                                └── testlesionmeasures.csv  