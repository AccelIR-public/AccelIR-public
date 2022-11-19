import numpy as np

import torch
import torch.nn as nn

def split_pil(image, patch_size, stride_size, thres_size):
    width, height = image.width, image.height

    h_space = np.arange(0, height - patch_size + 1, stride_size)
    if height - (h_space[-1] + patch_size) > thres_size:
        h_space = np.append(h_space, height - patch_size)
    w_space = np.arange(0, width - patch_size + 1, stride_size)
    if width - (w_space[-1] + patch_size) > thres_size:
        w_space = np.append(w_space, width - patch_size)

    crop_images = []
    for y in h_space:
        for x in w_space:
            crop_img = image.crop((x, y, x + patch_size, y + patch_size))
            crop_images += [crop_img]
    
    return crop_images

def calc_batch_psnr(img1, img2, device):
    psnr_list = []
    for i in range(len(img1)):
        single_img1 = img1[i].mul(255).clamp(0,255).round()
        single_img2 = img2[i].mul(255).clamp(0,255).round()

        mse = torch.mean((single_img1 - single_img2) ** 2)
        if torch.is_nonzero(mse):
            psnr = 20 * torch.log10(255.0 / torch.sqrt(mse))
        else:
            psnr = torch.tensor(100.0).to(device)
        psnr_list += [psnr]

    return torch.stack(psnr_list, dim=0)

def calc_batch_loss(img1, img2, loss_type):
    if loss_type == 'L1':
        criterion = nn.L1Loss()
    elif loss_type == 'L2':
        criterion = nn.MSELoss()
    loss_list = []
    for i in range(len(img1)):
        single_img1 = img1[i].mul(255).clamp(0,255).round()
        single_img2 = img2[i].mul(255).clamp(0,255).round()
        loss = criterion(single_img1, single_img2)
        loss_val = loss.data
        loss_list += [loss_val]
    return torch.stack(loss_list, dim=0)