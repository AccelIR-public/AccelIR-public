import os
import argparse

from tqdm import tqdm

from PIL import Image

from src.data.utility import split_pil

import glob

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    #directory, path
    parser.add_argument('--hr_data_dir', type=str, required=True)
    parser.add_argument('--lr_data_dir', type=str, required=True)
    parser.add_argument('--target_hr_data_dir', type=str, required=True)
    parser.add_argument('--target_lr_data_dir', type=str, required=True)

    # patch_size, scale
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument('--stride_size', type=int, default=28)
    parser.add_argument('--scale', type=int, default=4)

    # jpeg options
    parser.add_argument('--quality_step', type=int, default=1)

    args = parser.parse_args()

    # set up
    hr_image_files = sorted(glob.glob(os.path.join(args.hr_data_dir, '*.png')))
    lr_image_files = sorted(glob.glob(os.path.join(args.lr_data_dir, '*.png')))

    thres_size = args.patch_size//8 * 3

    os.makedirs(args.target_hr_data_dir, exist_ok=True)
    os.makedirs(args.target_lr_data_dir, exist_ok=True)

    # generate HR patch
    patch_size = args.patch_size * args.scale
    stride_size = args.stride_size * args.scale
    hr_thres_size = thres_size * args.scale
    for hr in tqdm(hr_image_files):
        base = os.path.basename(hr)
        name = base[:base.find('.')]
        target_hr_image_dir = os.path.join(args.target_hr_data_dir, name)
        os.makedirs(target_hr_image_dir, exist_ok=True)
        hr_image = Image.open(hr)
        hr_image.load()
        hr_crop_images = split_pil(hr_image, patch_size, stride_size, hr_thres_size)
        for i, hr_crop in enumerate(hr_crop_images):
            hr_crop.save(os.path.join(target_hr_image_dir, f'{i+1}.png'))

    # generate LR patch with JPEG encoding
    patch_size = args.patch_size
    stride_size = args.stride_size
    lr_thres_size = thres_size
    for lr in tqdm(lr_image_files):
        base = os.path.basename(lr)
        name = base[:base.find('.')]
        target_lr_image_dir = os.path.join(args.target_lr_data_dir, name)
        os.makedirs(target_lr_image_dir, exist_ok=True)
        lr_image = Image.open(lr)
        lr_image.load()
        lr_crop_images = split_pil(lr_image, patch_size, stride_size, lr_thres_size)
        for i, lr_crop in enumerate(lr_crop_images):
            os.makedirs(os.path.join(target_lr_image_dir, f'{i+1}', 'enc_images'), exist_ok=True)
            for q in range(0, 101, args.quality_step):
                if q == 0:
                    lr_crop.save(os.path.join(target_lr_image_dir, f'{i+1}', 'enc_images', f'{q}.png'))
                else:
                    lr_crop.save(os.path.join(target_lr_image_dir, f'{i+1}', 'enc_images', f'{q}.jpg'), quality=q)
