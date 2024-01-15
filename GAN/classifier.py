import torch
import torch.nn as nn
import torch.nn.functional as F

from GAN.utils import (
    GLU,
    GatedLinearUnit,
    PixelShuffle,
    FeatureSizeNormalizer,
    DownSample,
)


class Classifier(nn.Module):
    def __init__(self, cfg):
        super(Classifier, self).__init__()

        self.num_classes = len(cfg["DATASET"]["EMOTIONS"])
        self.num_mcep = cfg["INPUT"]["NUM_MCEP"]
        self.n_frames = cfg["INPUT"]["N_FRAMES"]

        self.conv1 = nn.Conv2d(
            1, 128, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1)
        )
        self.conv1_gates = nn.Conv2d(
            1, 128, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1)
        )
        self.glu1 = GatedLinearUnit()

        self.downsample1 = DownSample(
            128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)
        )
        self.downsample2 = DownSample(
            256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)
        )
        self.downsample3 = DownSample(
            512, 1024, kernel_size=(6, 3), stride=(1, 2), padding=(0, 1)
        )

        self.dense_emotion = nn.Linear(
            8 * self.num_mcep * (self.n_frames), self.num_classes
        )  # For emotion classification
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        # x shape: [batch, num_mcep, n_frames]
        x = x.unsqueeze(1)  # shape: [batch, 1, num_mcep, n_frames]

        h1 = self.leaky_relu(self.conv1(x))
        h1_gates = self.leaky_relu(self.conv1_gates(x))
        h1_glu = self.glu1(h1, h1_gates)

        d1 = self.downsample1(h1_glu)
        d2 = self.downsample2(d1)

        d3 = self.downsample3(d2)

        d3_flat = d3.view(d3.size(0), -1)
        emotion_output = self.dense_emotion(d3_flat)

        return emotion_output
