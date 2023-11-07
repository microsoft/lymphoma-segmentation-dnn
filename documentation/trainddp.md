# How to train a model using this codebase?

The models in this work are trained on a single-node with `torch.cuda.device_count()` GPUs. In our work, we had `torch.cuda.device_count() == 4` on a single Microsoft Azure VM (node). Each GPU consisted of 16 GiB of RAM. The machine consisted of 24 vCPUs and 448 GiB of RAM. 

Running a training experiment primarily uses only three files from this codebase: [config.py](./../config.py), [segmentation/trainddp.py](./../segmentation/trainddp.py) and [segmentation/initialize_train.py](./../segmentation/initialize_train.py). The first step is to initialize the correct values for the variable `LYMPHOMA_SEGMENTATION_FOLDER` in the [config.py](./../config.py). Put all the training (and test, if applicable) data inside the `LYMPHOMA_SEGMENTATION_FOLDER/data` folder, as described in [dataset_format.md](./dataset_format.md).

```
import os 

LYMPHOMA_SEGMENTATION_FOLDER = '/path/to/lymphoma.segmentation/folder/for/data/and/results' # path to the directory containing `data` and `results` (this will be created by the pipeline) folders.

DATA_FOLDER = os.path.join(LYMPHOMA_SEGMENTATION_FOLDER, 'data')
RESULTS_FOLDER = os.path.join(LYMPHOMA_SEGMENTATION_FOLDER, 'results')
os.makedirs(RESULTS_FOLDER, exist_ok=True)
WORKING_FOLDER = os.path.dirname(os.path.abspath(__file__))
```

If all the dataset is correctly configured based on the explanations in [dataset_format.md](./dataset_format.md) and the [config.py](./../config.py) is correctly initialized as well, you are all set to initiate the training script. 

## Step 1: Activate the required conda environment (`lymphoma_seg`) and navigate to `segmentation` folder
First, activate the conda environment `lymphoma_seg` using (created as described in [conda_env.md](./conda_env.md)):  

```
conda activate lymphoma_seg
cd segmentation
```

## Step 2: Run the training script
After this, run the following script in your terminal:  

```
torchrun --standalone --nproc_per_node=1 trainddp.py --fold=0 --network-name='unet' --epochs=500 --input-patch-size=192 --train-bs=1 --num_workers=2 --cache-rate=0.5 --lr=2e-4 --wd=1e-5 --val-interval=2 --sw-bs=2 
```

Here, we are using PyTorch's `torchrun` to start a multi-GPU training. The `standalone` represents that we are using just one node. 

- `--nproc_per_node` defines the number of processes per node; in this case it represents the number of GPUs you want to use to train your model. We used `--nproc_per_node=4`, but feel free to set this variable to the number of GPUs available in your machine. 

- `trainddp.py` is the file containing the code for training that uses `torch.nn.parallel.DistributedDataParallel`. 

- `--fold` defines the fold for which you want to run training. When the above script is run for the first time, two files, namely, `train_filepaths.csv` and `test_filepaths.csv` gets created within the folder `WORKING_FOLDER/data_split`, where the former contains the filepaths (CT, PT, mask) for training images (from `imagesTr` and `labelsTr` folders as described in `dataset_format.md`), and the latter contains the filepaths for test images (from `imagesTs` and `labelsTs`), respectively. The `train_filepaths.csv` contains a column named `FoldID` with values in `{0, 1, 2, 3, 4}` defining which fold the data in that row belongs to. When `--fold=0` (for example), the code uses all the data with `FoldID == 0` for validation and the data with `FoldID != 0` for training. Defaults to 0.

- `--network-name` defines the name of the network. In this work, we have trained UNet, SegResNet, DynUNet and SwinUNETR (adpated from MONAI [LINK]). Hence, the `--network-name` should be set to one of `unet`, `segresnet`, `dynunet`, or `swinunetr`. Defaults to `unet`.

- `--epochs` is the total number of epochs for running the training. Defaults to 500.

- `--input-patch-size` defines the size of the cubic input patch that is cropped from the input images during training. The code uses `monai.transforms.RRandCropByPosNegLabeld` (used inside `segmentation\initialize_train.py`) for creating these cropped patches. We used `input-patch-size` of 224 for UNet, 192 for SegResNet, 160 for DynUNet and 128 for SwinUNETR. Defaults to 192.

- `--train-bs` is the training batch size. We used `--train-bs = 1` for all our experiments in this work, since for the given `input-patch-size` for the networks above, we couldn't accommodate larger batch sizes for SegResNet, DynUNet, and SwinUNETR. Defaults to 1.

- `--num-workers` defines the `num_workers` argument inside training and validation DataLoaders. Defaults to 2.

- `--cache-rate` defines the precentage of cached data argument to be used inside the `monai.data.CacheDataset`. This type of dataset (unlike `torch.utils.data.Dataset`) can load and cache deterministic transforms result during training. A cache-rate of 1 caches all the data into the memory, while a cache-rate of 0 doesn't cache anything into the memory. A higher cache rate leads to faster training (but more memory consumption). Defaults to 0.1.

- `--lr` defines the initial learning rate. Cosine annealing scheduler is used to update the learning rate from the initial value to 0 in `epochs` epochs. Defaults to 2e-4.

- `--wd` defines the weight-decay for the AdamW optimizer used in this work. Defaults to 1e-5.

- `--val_interval` defines the interval for performing validation and saving the model being trained. Defaults to 2. 

- `--sw-bs` defines the batch size for performing the sliding-window inference via `monai.inferers.sliding_window_inference` on the validation inputs. Defaults to 2. 



Alternatively, modify the [segmentation/train.sh](./../segmentation/train.sh) script for your use-case (which contains the same bash script as above) and run:

```
bash train.sh
```