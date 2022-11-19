import torch
import torchvision.transforms.functional as TF

import numpy as np

class QNModule():    
    def __init__(self, qp_net, patch_size, device):
        self.qp_net = qp_net
        self.patch_size = patch_size
        self.device = device

    def _decompose(self, input_tensor):
        patch_size = self.patch_size
        width = input_tensor.shape[3]
        height = input_tensor.shape[2]
        h_splits = list(input_tensor.split(patch_size, dim=2))
        if not (h_splits[-1].shape[2] == patch_size):
            h_splits[-1] = input_tensor[:,:,height-patch_size:,:]
        for i, h_split in enumerate(h_splits):
            w_h_splits = list(h_split.split(patch_size, dim=3))
            if not (w_h_splits[-1].shape[3] == patch_size):
                w_h_splits[-1] = h_split[:,:,:,width-patch_size:]
            if i == 0:
                decomp_tensor = torch.cat(w_h_splits, dim=0)
            else:
                tmp_tensor = torch.cat(w_h_splits, dim=0)
                decomp_tensor = torch.cat([decomp_tensor, tmp_tensor], dim=0)
        
        return decomp_tensor

    def _generate_encoding_profile(self, np_pred, h, w):
        patch_size = self.patch_size

        width = int(w)
        height = int(h)

        w_space = np.arange(0, width, patch_size)
        h_space = np.arange(0, height, patch_size)

        assert(len(w_space) * len(h_space) == np_pred.shape[0])

        enc_profiles = []
        for y in range(len(h_space)):
            for x in range(len(w_space)):
                importance = np_pred[y * len(w_space) + x]
                enc_profile = [x*patch_size, y*patch_size, patch_size, int(importance)]
                enc_profiles += [enc_profile]
        return enc_profiles

    def infer_q(self, image):
        with torch.no_grad():
            self.qp_net.to(self.device)
            self.qp_net.eval()
            image_tensor = TF.to_tensor(image).unsqueeze(0).to(self.device)
            decomp_images = self._decompose(image_tensor)
            output = self.qp_net(decomp_images)
            importance = output.argmax(dim=1, keepdim=True).to('cpu')
            importance_np = importance.numpy()
            enc_profiles = self._generate_encoding_profile(importance_np, image_tensor.shape[2], image_tensor.shape[3])
        return enc_profiles
