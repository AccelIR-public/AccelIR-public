import torch.utils.data as data
import torchvision.transforms.functional as TF

import os

from PIL import Image
from tqdm import tqdm

import glob

class PatchDataset(data.Dataset):
    def __init__(self, args):
        self.args = args

        self.size_list = []

        self.hr_image_dir = os.path.join(self.args.hr_data_dir)
        self.lr_image_dir = os.path.join(self.args.lr_data_dir)

        self.hr_image_dirs = self._scan_dir(self.hr_image_dir)
        self.lr_image_dirs = self._scan_dir(self.lr_image_dir)

        self.hr_patch_files = self._scan_dirs(self.hr_image_dirs)
        self.lr_patch_dirs = self._scan_dirs(self.lr_image_dirs)

        # self.hr_patches = self._load_patches(self.hr_patch_files)
        # self.lr_patches = self._load_enc_patches(self.lr_patch_dirs)

    def _scan_dir(self, dir_path):
        return sorted(glob.glob(os.path.join(dir_path, '*')))

    def _scan_dirs(self, dir_path):
        file_list = []
        for direc in dir_path:
            files = self._scan_dir(direc)
            file_list += files
        
        return file_list

    def _load_patches(self, patch_files):
        patches = []

        for patch_path in tqdm(patch_files, desc='Loading patches...'):
            patch = Image.open(patch_path)
            patch.load()
            patches += [patch]
        
        return patches

    def _load_enc_patches(self, patch_dirs):
        patches = []

        for patch_dir in tqdm(patch_dirs, desc='Loading patches...'):
            patch_names = sorted(os.listdir(os.path.join(patch_dir, 'enc_images')), key=lambda x : int(x[:x.find('.')]))
            enc_patches = []
            enc_size_list = []
            for patch_name in patch_names:
                patch_file = os.path.join(patch_dir, 'enc_images', patch_name)
                patch = Image.open(patch_file)
                patch.load()
                enc_patches += [patch]
                size = os.path.getsize(patch_file)
                enc_size_list += [size]
            patches.append(enc_patches)
            self.size_list.append(enc_size_list)

        return patches

    def __getitem__(self, idx):
        hr_patch_path = self.hr_patch_files[idx]
        hr_patch = Image.open(hr_patch_path)
        hr_patch.load()
        
        lr_patch_dir = self.lr_patch_dirs[idx]
        lr_patch_names = sorted(os.listdir(os.path.join(lr_patch_dir, 'enc_images')), key=lambda x: int(x[:x.find('.')]))
        lr_patches = []
        sizes = []
        for lr_patch_name in lr_patch_names:
            lr_patch_path = os.path.join(lr_patch_dir, 'enc_images', lr_patch_name)
            lr_patch = Image.open(lr_patch_path)
            lr_patch.load()
            lr_patches.append(lr_patch)
            size = os.path.getsize(lr_patch_path)
            sizes.append(size)            

        hr_tensor = TF.to_tensor(hr_patch)
        lr_tensors = []
        for lr_patch in lr_patches:
            lr_tensors += [TF.to_tensor(lr_patch)]

        return hr_tensor, lr_tensors, sizes

    def __len__(self):
        return len(self.hr_patch_files)