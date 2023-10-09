#%%
'''
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
'''

from monai.transforms import (
    AsDiscrete,
    Compose,
)
import argparse
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, decollate_batch
import torch
import matplotlib.pyplot as plt
import os
import pandas as pd
import time
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist 
import os
from initialize_train import (
    create_data_split_files,
    get_train_valid_data_in_dict_format, 
    get_train_transforms, 
    get_valid_transforms, 
    get_model, 
    get_loss_function,
    get_optimizer, 
    get_scheduler,
    get_metric,
    get_validation_sliding_window_size
)

import sys
config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(config_dir)
from config import RESULTS_FOLDER
torch.backends.cudnn.benchmark = True
#%%
def ddp_setup():
    dist.init_process_group(backend='nccl', init_method="env://")

def convert_to_4digits(str_num):
    if len(str_num) == 1:
        new_num = '000' + str_num
    elif len(str_num) == 2:
        new_num = '00' + str_num
    elif len(str_num) == 3:
        new_num = '0' + str_num
    else:
        new_num = str_num
    return new_num

#%%
def load_train_objects(args):
    train_data, valid_data = get_train_valid_data_in_dict_format(args.fold) 
    train_transforms = get_train_transforms(args.input_patch_size)
    valid_transforms = get_valid_transforms()
    model = get_model(args.network_name, args.input_patch_size) 
    optimizer = get_optimizer(model, learning_rate=args.lr, weight_decay=args.wd)
    loss_function = get_loss_function()
    scheduler = get_scheduler(optimizer, args.epochs)
    metric = get_metric()

    return (
        train_data,
        valid_data,
        train_transforms,
        valid_transforms,
        model,
        loss_function,
        optimizer,
        scheduler,
        metric
    )


def prepare_dataset(data, transforms, args):
    dataset = CacheDataset(data=data, transform=transforms, cache_rate=args.cache_rate, num_workers=args.num_workers)
    return dataset


def main_worker(save_models_dir, save_logs_dir, args):
    # init_process_group
    ddp_setup() 
    # get local rank on the GPU
    local_rank = int(dist.get_rank())
    if local_rank == 0:
        print(f"Training {args.network_name} on fold {args.fold}")
        print(f"The models will be saved in {save_models_dir}")
        print(f"The training/validation logs will be saved in {save_logs_dir}")

    # get all training and validation objects
    train_data, valid_data, train_transforms, valid_transforms, model, loss_function, optimizer, scheduler, metric = load_train_objects(args)

    # get dataset of object-type CacheDataset 
    train_dataset = prepare_dataset(train_data, train_transforms, args)
    valid_dataset = prepare_dataset(valid_data, valid_transforms, args)

    # get DistributedSampler instances for both training and validation dataloader
    # this will be used to split data into different GPUs
    train_sampler = DistributedSampler(dataset=train_dataset, shuffle=True)
    valid_sampler = DistributedSampler(dataset=valid_dataset, shuffle=False)
    
    # initializing train and valid dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_bs,
        pin_memory=True,
        shuffle=False,
        sampler=train_sampler,
        num_workers=args.num_workers
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=1,
        pin_memory=True,
        shuffle=False,
        sampler=valid_sampler,
        num_workers=args.num_workers
    )

    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
    post_label = Compose([AsDiscrete(to_onehot=2)])

    # filepaths for storing training and validation logs from different GPUs
    trainlog_fpath = os.path.join(save_logs_dir, f'trainlog_gpu{local_rank}.csv')
    validlog_fpath = os.path.join(save_logs_dir, f'validlog_gpu{local_rank}.csv')

    # initialize the GPU device    
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # number of epochs and epoch interval for running validation
    max_epochs = args.epochs
    val_interval = args.val_interval

    # push models to device
    model = model.to(device)

    epoch_loss_values = []
    metric_values = []

    # wrap the model with DDP
    model = DDP(model, device_ids=[device])
        
    experiment_start_time = time.time()
    
    for epoch in range(max_epochs):
        epoch_start_time = time.time()
        print(f"[GPU{local_rank}]: Running training: epoch = {epoch + 1}")
        model.train()
        epoch_loss = 0
        step = 0
        train_sampler.set_epoch(epoch)
        for batch_data in train_dataloader:
            step += 1
            inputs, labels = (
                batch_data['CTPT'].to(device),
                batch_data['GT'].to(device),
            )
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= step
        print(f"[GPU:{local_rank}]: epoch {epoch + 1}/{max_epochs}: average loss: {epoch_loss:.4f}")
        epoch_loss_values.append(epoch_loss)

        # steps forward the CosineAnnealingLR scheduler
        scheduler.step()

        # update the training log file
        epoch_loss_values_df = pd.DataFrame(data=epoch_loss_values, columns=['Loss'])
        epoch_loss_values_df.to_csv(trainlog_fpath, index=False)


        if (epoch + 1) % val_interval == 0:
            print(f"[GPU{local_rank}]: Running validation")
            model.eval()
            with torch.no_grad():
                for val_data in valid_dataloader:
                    val_inputs, val_labels = (
                        val_data['CTPT'].to(device),
                        val_data['GT'].to(device),
                    )
                    roi_size = get_validation_sliding_window_size(args.input_patch_size) 
                    sw_batch_size = args.sw_bs
                    val_outputs = sliding_window_inference(
                        val_inputs, roi_size, sw_batch_size, model)
                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                    # compute metric for current iteration
                    metric(y_pred=val_outputs, y=val_labels)

                # aggregate the final mean dice result
                metric_val = metric.aggregate().item()
                metric.reset()
                metric_values.append(metric_val)
                metric_values_df = pd.DataFrame(data=metric_values, columns=['Metric'])
                metric_values_df.to_csv(validlog_fpath, index=False)
               
                print(f"[GPU:{local_rank}] SAVING MODEL at epoch: {epoch + 1}; Mean DSC: {metric_val:.4f}")
                savepath = os.path.join(save_models_dir, "model_ep="+convert_to_4digits(str(int(epoch + 1)))+".pth")
                torch.save(model.module.state_dict(), savepath)

        epoch_end_time = (time.time() - epoch_start_time)/60
        print(f"[GPU:{local_rank}]: Epoch {epoch + 1} time: {round(epoch_end_time,2)} min")
       
    experiment_end_time = (time.time() - experiment_start_time)/(60*60)
    print(f"[GPU:{local_rank}]: Total time: {round(experiment_end_time,2)} hr")

    dist.destroy_process_group()

def main(args):
    os.environ['OMP_NUM_THREADS'] = '6'
    fold = args.fold
    network = args.network_name
    inputsize = f'randcrop{args.input_patch_size}'

    # extrafeature = 'petnotnormalized_p2'
    experiment_code = f"{network}_fold{fold}_{inputsize}"

    #save models folder
    save_models_dir = os.path.join(RESULTS_FOLDER,'models')
    save_models_dir = os.path.join(save_models_dir, 'fold'+str(fold), network, experiment_code)
    os.makedirs(save_models_dir, exist_ok=True)
    
    # save train and valid logs folder
    save_logs_dir = os.path.join(RESULTS_FOLDER,'logs')
    save_logs_dir = os.path.join(save_logs_dir, 'fold'+str(fold), network, experiment_code)
    os.makedirs(save_logs_dir, exist_ok=True)
    
    main_worker(save_models_dir, save_logs_dir, args)
    


if __name__ == "__main__": 
    # create datasplit files for train and test images
    # follow all the instructions for dataset directory creation and images/labels file names as given in: LINK
    create_data_split_files() 
    parser = argparse.ArgumentParser(description='Lymphoma PET/CT lesion segmentation using MONAI-PyTorch')
    parser.add_argument('--fold', type=int, default=0, metavar='fold',
                        help='validation fold (default: 0), remaining folds will be used for training')
    parser.add_argument('--network-name', type=str, default='unet', metavar='netname',
                        help='network name for training (default: unet)')
    parser.add_argument('--epochs', type=int, default=500, metavar='epochs',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--input-patch-size', type=int, default=192, metavar='inputsize',
                        help='size of cropped input patch for training (default: 192)')
    parser.add_argument('--train-bs', type=int, default=1, metavar='train-bs',
                        help='mini-batchsize for training (default: 1)')
    parser.add_argument('--num_workers', type=int, default=2, metavar='nw',
                        help='num_workers for train and validation dataloaders (default: 2)')
    parser.add_argument('--cache-rate', type=float, default=0.1, metavar='cr',
                        help='cache_rate for CacheDataset from MONAI (default=0.1)')
    parser.add_argument('--lr', type=float, default=2e-4, metavar='lr',
                        help='initial learning rate for AdamW optimizer (default=2e-4); Cosine scheduler will decrease this to 0 in args.epochs epochs')
    parser.add_argument('--wd', type=float, default=1e-5, metavar='wd',
                        help='weight-decay for AdamW optimizer (default=1e-5)')
    parser.add_argument('--val-interval', type=int, default=2, metavar='val-interval',
                        help='epochs interval for which validation will be performed (default=2)')
    parser.add_argument('--sw-bs', type=int, default=2, metavar='sw-bs',
                        help='batchsize for sliding window inference (default=2)')
    args = parser.parse_args()
    
    main(args)

