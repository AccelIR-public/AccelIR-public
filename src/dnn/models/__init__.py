from .edsr import EDSR
from .carn import CARN_M
from .qp_net import QPNET
from .swinir import SwinIR

def build_sr(model_name, input_channels, output_channels, num_channels, num_blocks, scale):
    if model_name == 'edsr':
        model = EDSR(input_channels=input_channels, output_channels=output_channels, num_channels=num_channels, upscale=scale)
    elif model_name == 'carn':
        model = CARN_M(in_nc=3, out_nc=3, nf=num_channels, scale=scale)
    elif model_name == 'swinir':
        model = SwinIR(upscale=scale, in_chans=3, img_size=32, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6], embed_dim=num_channels, num_heads=[6, 6, 6, 6],
                        mlp_ratio=2, upsampler='pixelshuffledirect', resi_connection='1conv')
    else:
        raise NotImplementedError

    return model

def build_qpnet(num_layers, num_channels, output_dim, patch_size, sr_model_name):
    model = QPNET(num_layers=num_layers, num_channels=num_channels, output_dim=output_dim, patch_size=patch_size, sr_model_name=sr_model_name)

    return model