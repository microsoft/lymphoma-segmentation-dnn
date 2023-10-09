#%%
'''
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
'''
import numpy as np 
import glob
import os 
import pandas as pd 
import SimpleITK as sitk
import sys
import argparse
from monai.inferers import sliding_window_inference
from monai.data import DataLoader, Dataset, decollate_batch
import torch
import os
import glob
import pandas as pd
import numpy as np
import torch.nn as nn
import time
from initialize_train import (
    get_validation_sliding_window_size,
    get_model,
    get_test_data_in_dict_format,
    get_valid_transforms,
    get_post_transforms
)
config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(config_dir)
from config import RESULTS_FOLDER
#%%
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

def create_dictionary_ctptgt(ctpaths, ptpaths, gtpaths):
    data = []
    for i in range(len(gtpaths)):
        ctpath = ctpaths[i]
        ptpath = ptpaths[i]
        gtpath = gtpaths[i]
        data.append({'CT':ctpath, 'PT':ptpath, 'GT':gtpath})
    return data

def read_image_array(path):
    img =  sitk.ReadImage(path)
    array = np.transpose(sitk.GetArrayFromImage(img), (2,1,0))
    return array

#%%
def main(args):
    # initialize inference
    fold = args.fold
    network = args.network_name
    inputsize = args.input_patch_size
    experiment_code = f"{network}_fold{fold}_randcrop{inputsize}"
    sw_roi_size = get_validation_sliding_window_size(inputsize) # get sliding_window inference size for given input patch size
    
    # find the best model for this experiment from the training/validation logs
    # best model is the model with the best validation `Metric` (DSC)
    save_logs_dir = os.path.join(RESULTS_FOLDER, 'logs')
    validlog_fname = os.path.join(save_logs_dir, 'fold'+str(fold), network, experiment_code, 'validlog_gpu0.csv')
    validlog = pd.read_csv(validlog_fname)
    best_epoch = 2*(np.argmax(validlog['Metric']) + 1)
    best_metric = np.max(validlog['Metric'])
    print(f"Using the {network} model at epoch={best_epoch} with mean valid DSC = {round(best_metric, 4)}")

    # get the best model and push it to device=cuda:0
    save_models_dir = os.path.join(RESULTS_FOLDER,'models')
    save_models_dir = os.path.join(save_models_dir, 'fold'+str(fold), network, experiment_code)
    best_model_fname = 'model_ep=' + convert_to_4digits(str(best_epoch)) +'.pth'
    model_path = os.path.join(save_models_dir, best_model_fname)
    device = torch.device(f"cuda:0")
    model = get_model(network, input_patch_size=inputsize)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
        
    # initialize the location to save predicted masks
    save_preds_dir = os.path.join(RESULTS_FOLDER, f'predictions')
    save_preds_dir = os.path.join(save_preds_dir, 'fold'+str(fold), network, experiment_code)
    os.makedirs(save_preds_dir, exist_ok=True)

    # get test data (in dictionary format for MONAI dataloader), test_transforms and post_transforms
    test_data = get_test_data_in_dict_format()
    test_transforms = get_valid_transforms()
    post_transforms = get_post_transforms(test_transforms, save_preds_dir)
    
    # initalize PyTorch dataset and Dataloader
    dataset_test = Dataset(data=test_data, transform=test_transforms)
    dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=args.num_workers)

    model.eval()
    with torch.no_grad():
        for data in dataloader_test:
            inputs = data['CTPT'].to(device)
            sw_batch_size = args.sw_bs
            print(sw_batch_size)
            data['Pred'] = sliding_window_inference(inputs, sw_roi_size, sw_batch_size, model)
            data = [post_transforms(i) for i in decollate_batch(data)]


if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Lymphoma PET/CT lesion segmentation using MONAI-PyTorch')
    parser.add_argument('--fold', type=int, default=0, metavar='fold',
                        help='validation fold (default: 0), remaining folds will be used for training')
    parser.add_argument('--network-name', type=str, default='unet', metavar='netname',
                        help='network name for training (default: unet)')
    parser.add_argument('--input-patch-size', type=int, default=192, metavar='inputsize',
                        help='size of cropped input patch for training (default: 192)')
    parser.add_argument('--num_workers', type=int, default=2, metavar='nw',
                        help='num_workers for train and validation dataloaders (default: 2)')
    parser.add_argument('--sw-bs', type=int, default=2, metavar='sw-bs',
                        help='batchsize for sliding window inference (default=2)')
    args = parser.parse_args()
    
    main(args)

