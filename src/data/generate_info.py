import os
import re
from PIL import Image

import argparse

from src.data.utility import calc_batch_loss, calc_batch_psnr
from src.data.patch_dataset import PatchDataset
from src.dnn.models import build_sr

import numpy as np
import pandas as pd

from tqdm import tqdm

import torch

import glob

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #directory, path
    parser.add_argument('--hr_data_dir', type=str, required=True)
    parser.add_argument('--lr_data_dir', type=str, required=True)

    # patch_size, scale
    parser.add_argument('--scale', type=int, default=4)

    # dnn option
    parser.add_argument('--model_name', type=str, default='edsr')
    parser.add_argument('--num_channels', type=int, default=32)
    parser.add_argument('--num_blocks', type=int, default=8)
    
    parser.add_argument('--checkpoint_dir', type=str, default='./pretrained')

    parser.add_argument('--use_cpu', action='store_true')
    parser.add_argument('--device', type=int, default=0)

    # jpeg options
    parser.add_argument('--quality_step', type=int, default=5)

    args = parser.parse_args()
    
    # set torch device
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.device}'
    device = torch.device('cpu' if args.use_cpu else 'cuda')

    # load patch dataset
    patch_dataset = PatchDataset(args)

    # load sr model
    model = build_sr(model_name=args.model_name, input_channels=3, output_channels=3, num_channels=args.num_channels, num_blocks=args.num_blocks, scale=args.scale)
    
    if args.model_name == 'swinir':
        checkpoint_path = os.path.join(args.checkpoint_dir, f'SwinIR_F{args.num_channels}.pth')
    else:
        checkpoint_path = os.path.join(args.checkpoint_dir, f'{model.name}.pth')

    checkpoint_path = os.path.join(args.checkpoint_dir, f'{model.name}.pth')

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    model.to(device)
    
    start_q = 5

    with torch.no_grad():
        for i in tqdm(range(len(patch_dataset)), desc='Getting info...'):
            hr_tensor, lr_tensors, size_list = patch_dataset[i]
            base_path = patch_dataset.lr_patch_dirs[i]

            # make as input
            input = torch.stack(lr_tensors[start_q:-1:args.quality_step], dim=0)
            target = torch.stack([hr_tensor for _ in range(len(lr_tensors[start_q:-1:args.quality_step]))], dim=0)

            input, target = input.to(device), target.to(device)

            output = model(input)

            psnr = calc_batch_psnr(output, target, device)
            l1 = calc_batch_loss(output, target, "L1")
            l2 = calc_batch_loss(output, target, "L2")

            qual_np = np.arange(start_q, 100, args.quality_step)
            psnr_np = psnr.to('cpu').numpy()
            l1_np = l1.to('cpu').numpy()
            l2_np = l2.to('cpu').numpy()
            size_np = np.array(size_list[start_q:-1:args.quality_step])

            log_dir = os.path.join(base_path, 'log')
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, f'size_qual_info_{args.model_name}_{args.num_channels}.txt')
            
            df = pd.DataFrame(columns=['quality', 'size', 'psnr', 'l1', 'l2'])
            df['quality'] = qual_np
            df['size'] = size_np
            df['psnr'] = psnr_np
            df['l1'] = l1_np
            df['l2'] = l2_np
            df.to_csv(log_path, sep='\t', index=False)
