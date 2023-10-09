#%%
'''
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
'''
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

def plot_image_mask_superposed_coronalmip(ptarray, maskarray, ax=None):
    pt_cor = np.rot90(np.max(ptarray, axis=1))
    mask_cor = np.rot90(np.max(maskarray, axis=1))
    ax.imshow(pt_cor, cmap='Greys')
    mask_cor_alpha = np.zeros_like(mask_cor, dtype=np.float32)
    mask_cor_alpha[mask_cor == 1] = 1  # Set alpha value for mask regions to 0.6
    ax.imshow(mask_cor, cmap='coolwarm', alpha=mask_cor_alpha)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax
