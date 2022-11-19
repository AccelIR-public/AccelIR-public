import torch

import torchvision.transforms.functional as TF

def split_input(pil_image, split_by):
    image_list = []
    width = pil_image.width
    height = pil_image.height
    step_width = width//split_by
    for split in range(split_by):
        crop_image = pil_image.crop((0 + split*step_width, 0, 0 + (split+1)*step_width, height))
        image_list.append(crop_image)
    return image_list

def concat_output(tensor_output_list):
    for i, tensor_output in enumerate(tensor_output_list):
        if i == 0:
            concat_output = tensor_output
        else:
            concat_output = torch.cat([concat_output, tensor_output], dim=3)
    return concat_output

def sr_inference(model_list, input_image, split_by, device):
    sr_tensor_list = []
    for model in model_list:
        with torch.no_grad():
            if split_by != 1:
                split_images = split_input(input_image, split_by)
                split_outputs = []
                for split_image in split_images:
                    split_tensor = TF.to_tensor(split_image).unsqueeze(dim=0)
                    split_tensor = split_tensor.to(device)
                    split_output = model(split_tensor)
                    split_output = split_output.to('cpu')
                    split_outputs.append(split_output)

                sr_tensor = concat_output(split_outputs)
            else:
                input_tensor = TF.to_tensor(input_image).unsqueeze(0).to(device)
                sr_tensor = model(input_tensor)
                sr_tensor = sr_tensor.to('cpu')
            sr_tensor_list.append(sr_tensor)

    return sr_tensor_list

def calc_psnr(img1, img2):
    img1 = img1.mul(255).clamp(0,255).round()
    img2 = img2.mul(255).clamp(0,255).round()

    mse = torch.mean((img1 - img2) ** 2)
    if torch.is_nonzero(mse):
        psnr = 20 * torch.log10(255.0 / torch.sqrt(mse))
    else:
        psnr = torch.tensor(100.0)

    return psnr

def tensor2pil(tensor_image, mode='RGB'):
    if len(tensor_image.size()) == 4 and tensor_image.size()[0] == 1:
        tensor_image = tensor_image.view(tensor_image.size()[1:])
        tensor_image = tensor_image.cpu()
    else:
        raise Exception("shape of image tensor must be single batch")
    
    return TF.to_pil_image(tensor_image.detach(), mode=mode)
    