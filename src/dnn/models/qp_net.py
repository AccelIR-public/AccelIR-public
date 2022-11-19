import torch.nn as nn

class QPNET(nn.Module):
    def __init__(self, num_layers, num_channels, output_dim, patch_size, sr_model_name):
        super(QPNET, self).__init__()

        output_dim = output_dim
        num_channels = num_channels
        num_layers = num_layers
        self.name = f'QPNET_L{num_layers}_F{num_channels}_O{output_dim}_{patch_size}_{sr_model_name}'
        self.lastOut_q = nn.Linear(128, output_dim)

        layers = []
        layers.append(nn.Conv2d(3, num_channels, 4, 4))
        layers.append(nn.ReLU(True))

        for _ in range(num_layers):
            layers.append(nn.Conv2d(num_channels, num_channels, 1))
            layers.append(nn.ReLU(True))
        self.CondNet = nn.Sequential(*layers)

        self.path_q = nn.Conv2d(num_channels, 128, 1)

    def forward(self, x):
        feat = self.CondNet(x)

        out_q = self.path_q(feat)
        out_q = nn.AvgPool2d(int(out_q.size()[2]))(out_q)
        out_q = out_q.view(out_q.size(0), -1)
        out_q = self.lastOut_q(out_q)

        return out_q