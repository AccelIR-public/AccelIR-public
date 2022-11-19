import torch.utils.data as data
import torchvision.transforms.functional as TF
import torch

import os

from PIL import Image
from tqdm import tqdm

import pandas as pd

import numpy as np
import random

from src.dnn.data.utility import get_slope

import glob

class QPatchDataset(data.Dataset):
    def __init__(self, args, is_train, threshold_list=None):
        self.args = args
        self.is_train = is_train

        # load lr patches (as input)
        if is_train:
            self.lr_image_dir = os.path.join(args.data_dir, 'patch', 'DIV2K_train_LR_bicubic', f'X{args.scale}')
        else:
            self.lr_image_dir = os.path.join(args.data_dir, 'patch', 'DIV2K_valid_LR_bicubic', f'X{args.scale}')
            
        self.lr_image_dirs = self._scan_dir(self.lr_image_dir)
        self.lr_patch_dirs = self._scan_dirs(self.lr_image_dirs)
        self.lr_patches = self._load_input_patches(self.lr_patch_dirs)

        # load slope info
        self.lr_slopes, self.size_qual_df = self._load_slop_info(self.lr_patch_dirs)
        
        # remove outliers
        self.lr_patches, self.lr_slopes = self._remove_outliers(self.lr_patches, self.lr_slopes)

        # make target (as output)
        if is_train:
            self.target_class, self.threshold_list = self._make_target(self.lr_slopes)
            self.size_qual_df['importance'] = np.repeat(self.target_class, 19)
            self.size_qual_df['size'] = self.size_qual_df['size'] - 625
        else:
            self.threshold_list = threshold_list
            self.target_class = self._make_target_test(self.lr_slopes, self.threshold_list)
            
    def _load_empty_patches(self, patch_dirs):
        patches = []

        for patch_dir in patch_dirs:
            patches += [0]
        return patches

    def _scan_dir(self, dir_path):
        return sorted(glob.glob(os.path.join(dir_path, '*')))

    def _scan_dirs(self, dir_path):
        file_list = []
        for direc in dir_path:
            files = self._scan_dir(direc)
            file_list += files
        
        return file_list

    def _load_input_patches(self, patch_dirs):
        patches = []

        for patch_dir in tqdm(patch_dirs, desc='Loading patches...'):
            patch_path = os.path.join(patch_dir, 'enc_images', '0.png')
            patch = Image.open(patch_path)
            patch.load()
            patches += [patch]

        return patches

    def _load_slop_info(self, log_dirs):
        slopes = []
        if self.args.mode == 'psnr':
            idx = 0
        elif self.args.mode == 'l1':
            idx = 1
        elif self.args.mode == 'l2':
            idx = 2
            
        for i, log_dir in enumerate(tqdm(log_dirs, desc='Loading slopes...')):
            log_path = os.path.join(log_dir, 'log', 'size_qual_info.txt')
            log_df = pd.read_csv(log_path, sep="\t")
            slope = get_slope(log_df, 20, 95, idx)
            slopes += [slope]
            if i == 0:
                size_qual_df = log_df
            else:
                size_qual_df = pd.concat([size_qual_df, log_df], axis=0)
        return slopes, size_qual_df

    def _remove_outliers(self, patches, slopes):
        np_slopes = np.array(slopes)
        np_patches = np.array(patches)
        indexes = np.where(np_slopes != 999999)[0]
        
        np_slopes = np_slopes[indexes]
        np_patches = np_patches[indexes]
        
        return np_patches.tolist(), np_slopes.tolist()

    def _make_target(self, slopes):
        mode = self.args.mode
        output_dim = self.args.output_dim
        # make rank list
        np_slopes = np.array(slopes)
        sort_index = np.argsort(np_slopes)
        if mode == 'psnr':
            sort_index = sort_index[::-1]
        
        # divide by desired dimension 
        target_output = np.zeros(len(slopes))
        step_size = len(slopes)//output_dim
        base = 0
        threshold_list = []
        for rank in range(output_dim):
            if rank == output_dim - 1:
                target_output[sort_index[base:]] = rank
                threshold = np_slopes[sort_index[-1]]
            else:
                target_output[sort_index[base: base+step_size]] = rank
                threshold = np_slopes[sort_index[base+step_size]]
            base += step_size
            threshold_list.append(threshold)
        
        return target_output, threshold_list

    def _make_target_test(self, slopes, threshold_list):
        mode = self.args.mode
        output_dim = self.args.output_dim
        
        assert(len(threshold_list) == output_dim)
        
        # make rank list
        np_slopes = np.array(slopes)
        sort_index = np.argsort(np_slopes)
        if mode == 'psnr':
            sort_index = sort_index[::-1]
        
        # divide by desired dimension 
        target_output = np.zeros(len(slopes))
        base = 0
        
        for i in range(output_dim):
            if i == 0:
                target_output[np.where((np_slopes < threshold_list[i]) & (np_slopes > -9999999))[0]] = i
            elif i == output_dim - 1:
                target_output[np.where((np_slopes < 9999999) & (np_slopes > threshold_list[i-1]))[0]] = i
            else:
                target_output[np.where((np_slopes < threshold_list[i]) & (np_slopes > threshold_list[i-1]))[0]] = i
        
        return target_output

    def _get_index(self, idx):
        if self.is_train:
            return idx
        else:
            return random.randint(0, len(self.lr_patches)-1)
    
    def __getitem__(self, idx):
        index = self._get_index(idx)
        lr_patch = self.lr_patches[index]
        target_class = self.target_class[index]

        lr_tensor = TF.to_tensor(lr_patch)
        target_tensor = torch.from_numpy(target_class.reshape(-1)).long().squeeze()

        return lr_tensor, target_tensor

    def __len__(self):
        if self.is_train:
            return len(self.lr_patches)
        else:
            return 1000


    
    


