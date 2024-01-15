import torch.nn as nn
import torch
import numpy as np

from GAN.utils import (
    DownSample,
    ResidualLayer,
    GLU,
    GatedLinearUnit,
    PixelShuffle,
    FeatureSizeNormalizer,
)


class Discriminator(nn.Module):
    def __init__(self, cfg):
        super(Discriminator, self).__init__()

        self.num_classes = len(cfg["DATASET"]["EMOTIONS"])
        self.num_mcep = cfg["INPUT"]["NUM_MCEP"]
        self.n_frames = cfg["INPUT"]["N_FRAMES"]

        # self.conv1 = nn.Conv2d(1, 128, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1))
        # self.conv1_gates = nn.Conv2d(1, 128, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1))
        # self.glu1 = GatedLinearUnit()

        self.downsample0 = DownSample(
            1, 128, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1)
        )
        self.downsample1 = DownSample(
            128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)
        )
        self.downsample2 = DownSample(
            256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)
        )
        self.downsample3 = DownSample(
            512, 1024, kernel_size=(6, 3), stride=(1, 2), padding=(0, 1)
        )

        # encoder_block(1, 64, self.gated)

        self.dense_real_fake = nn.Linear(
            8 * self.num_mcep * (self.n_frames), 1
        )  # For real/fake classification
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        # x shape: [batch, num_mcep, n_frames]
        x = x.unsqueeze(1)  # shape: [batch, 1, num_mcep, n_frames]

        # h1 = self.leaky_relu(self.conv1(x))
        # h1_gates = self.leaky_relu(self.conv1_gates(x))
        # h1_glu = self.glu1(h1, h1_gates)

        d0 = self.downsample0(x)
        d1 = self.downsample1(d0)
        d2 = self.downsample2(d1)
        intermediate_features = d2.view(d2.size(0), -1)

        d3 = self.downsample3(d2)

        d3_flat = d3.view(d3.size(0), -1)

        real_fake_output = torch.sigmoid(self.dense_real_fake(d3_flat))

        return real_fake_output, intermediate_features
