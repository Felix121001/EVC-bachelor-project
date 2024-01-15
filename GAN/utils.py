import torch
import torch.nn as nn
import torch.nn.functional as F


class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()
        # Custom Implementation because the Voice Conversion Cycle GAN
        # paper assumes GLU won't reduce the dimension of tensor by 2.

    def forward(self, input):
        return input * torch.sigmoid(input)


class GatedLinearUnit(nn.Module):
    def __init__(self):
        super(GatedLinearUnit, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, gates):
        return x * self.sigmoid(gates)


class PixelShuffle(nn.Module):
    def __init__(self, upscale_factor):
        super(PixelShuffle, self).__init__()
        # Custom Implementation because PyTorch PixelShuffle requires,
        # 4D input. Whereas, in this case we have have 3D array
        self.upscale_factor = upscale_factor

    def forward(self, input):
        n = input.shape[0]
        c_out = input.shape[1] // 2
        w_new = input.shape[2] * 2
        return input.view(n, c_out, w_new)


class FeatureSizeNormalizer(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)


class ResidualLayer(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, num_classes=0
    ):
        super(ResidualLayer, self).__init__()
        self.num_classes = num_classes

        self.conv1d_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels + num_classes,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
            ),
            nn.BatchNorm2d(num_features=out_channels, affine=True),
        )

        self.conv_layer_gates = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels + num_classes,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
            ),
            nn.BatchNorm2d(num_features=out_channels, affine=True),
        )

        self.conv1d_out_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=in_channels + num_classes,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
            ),
            nn.BatchNorm2d(num_features=in_channels + num_classes, affine=True),
        )

    def forward(self, input):
        h1_norm = self.conv1d_layer(input)
        h1_gates_norm = self.conv_layer_gates(input)

        # GLU
        h1_glu = h1_norm * torch.sigmoid(h1_gates_norm)

        h2_norm = self.conv1d_out_layer(h1_glu)
        # print("input.shape", input.shape, "h2_norm.shape", h2_norm.shape)
        return (input + h2_norm)[:, : -self.num_classes]


class ResidualLayer_combined(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, num_classes=0
    ):
        super(ResidualLayer_combined, self).__init__()

        self.conv1d_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
            ),
            nn.BatchNorm2d(num_features=out_channels, affine=True),
        )

        self.conv_layer_gates = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels + num_classes,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
            ),
            nn.BatchNorm2d(num_features=out_channels, affine=True),
        )

        self.conv1d_out_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
            ),
            nn.BatchNorm2d(num_features=in_channels, affine=True),
        )

    def forward(self, input1, input2):
        h1_norm = self.conv1d_layer(input1)
        h1_gates_norm = self.conv_layer_gates(input2)

        h1_glu = h1_norm * torch.sigmoid(h1_gates_norm)

        h2_norm = self.conv1d_out_layer(h1_glu)
        return input1 + h2_norm


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(DownSample, self).__init__()

        self.convLayer = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.InstanceNorm2d(num_features=out_channels, affine=True),
        )
        self.convLayer_gates = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.InstanceNorm2d(num_features=out_channels, affine=True),
        )

    def forward(self, input):
        # GLU
        return self.convLayer(input) * torch.sigmoid(self.convLayer_gates(input))


def upSample(in_channels, out_channels, kernel_size, stride, padding, num_classes=0):
    conv_out_channels = out_channels * 4
    convLayer = nn.Sequential(
        nn.Conv2d(
            in_channels=in_channels + num_classes,
            out_channels=conv_out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        ),
        nn.PixelShuffle(upscale_factor=2),
        nn.InstanceNorm2d(num_features=out_channels, affine=True),
        GLU(),
    )
    return convLayer


def upSample1D(
    in_channels,
    out_channels,
    kernel_size,
    stride,
    padding,
    output_padding=0,
    num_classes=0,
):
    # Upsample the feature maps using ConvTranspose1d
    # Setting stride to 2 to double the output length
    upsampleLayer = nn.ConvTranspose1d(
        in_channels=in_channels + num_classes,
        out_channels=out_channels,  # Directly output the desired number of channels
        kernel_size=kernel_size,
        stride=2,  # Stride set to 2 to double the sequence length
        padding=padding,
        output_padding=output_padding,
    )  # Adjust output padding if needed

    # Instance normalization and GLU activation
    normLayer = nn.InstanceNorm1d(num_features=out_channels, affine=True)
    activationLayer = GLU()

    # Combine the layers into a Sequential module
    convLayer = nn.Sequential(upsampleLayer, normLayer, activationLayer)

    return convLayer


class conv_block_gated(nn.Module):
    def __init__(self, in_c, out_c, num_classes=0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c + num_classes, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(in_c + num_classes, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        return self.bn1(self.conv1(inputs)) * torch.sigmoid(
            self.bn2(self.conv2(inputs))
        )


class conv_block_gated_1D(nn.Module):
    def __init__(self, in_c, out_c, num_classes=0):
        super().__init__()
        self.conv1 = nn.Conv1d(in_c + num_classes, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_c)
        self.conv2 = nn.Conv1d(in_c + num_classes, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_c)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        return self.bn1(self.conv1(inputs)) * torch.sigmoid(
            self.bn2(self.conv2(inputs))
        )


class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class encoder_block(nn.Module):
    def __init__(self, in_c, out_c, gated=False):
        super().__init__()
        self.conv = conv_block_gated(in_c, out_c) if gated else conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p


class encoder_block_1D(nn.Module):
    def __init__(self, in_c, out_c, gated=False):
        super().__init__()
        self.conv = (
            conv_block_gated_1D(in_c, out_c) if gated else conv_block(in_c, out_c)
        )
        self.pool = nn.MaxPool1d(2)

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p


class decoder_block(nn.Module):
    def __init__(self, in_c, out_c, padding=0, gated=False, num_classes=0):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_c, out_c, kernel_size=2, stride=2, output_padding=(padding, 0)
        )
        self.conv = (
            conv_block_gated(out_c + out_c + num_classes, out_c)
            if gated
            else conv_block(out_c + out_c + num_classes, out_c)
        )

    def forward(self, inputs, skip):
        x = self.up(inputs)
        # print("x.shape", x.shape, "skip.shape", skip.shape)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x


class decoder_block_1D(nn.Module):
    def __init__(self, in_c, out_c, padding=0, gated=False, num_classes=0):
        super().__init__()
        self.up = nn.ConvTranspose1d(
            in_c, out_c, kernel_size=2, stride=2, output_padding=padding
        )
        self.conv = (
            conv_block_gated_1D(out_c + out_c + num_classes, out_c)
            if gated
            else conv_block(out_c + out_c + num_classes, out_c)
        )

    def forward(self, inputs, skip):
        x = self.up(inputs)
        # print("x.shape", x.shape, "skip.shape", skip.shape)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x


import matplotlib.pyplot as plt
import librosa.display
import librosa


def visualize_f0(f0, sampling_rate, title="F0 Plot"):
    plt.figure(figsize=(10, 4))
    times = [i / sampling_rate for i in range(len(f0))]
    plt.plot(times, f0)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.grid(True)
    plt.show()


def visualize_coded_spectrogram(coded_sp, sr, title="Coded Spectrogram"):
    if coded_sp.ndim == 3:
        coded_sp = coded_sp[0, 0]

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(coded_sp, sr=sr, x_axis="time", y_axis="mel")
    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
    plt.show()
    
    
    
def safe_coded_spectrogram(coded_sp, sr, title="Spectrogram", filename="spectrogram.png"):
    if coded_sp.ndim == 3:
        coded_sp = coded_sp[0]

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(coded_sp, sr=sr, x_axis="time", y_axis="mel")
    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
    plt.savefig(filename)  
    plt.show()
    plt.close() 


def save_2d_vector_as_image(vector_2d, title="2D Vector Visualization", filename="2d_vector.png"):
    if vector_2d.ndim == 3:
        vector_2d = vector_2d[0, 0]

    plt.figure(figsize=(10, 4))
    plt.imshow(vector_2d, aspect='auto', interpolation='nearest')
    plt.colorbar()
    plt.title(title)
    plt.savefig(filename)
    plt.close()
    

import matplotlib.pyplot as plt
import librosa.display
import librosa
import numpy as np


def safe_spectrogram(sp, sr, title="Coded Spectrogram", filename="spectrogram.png"):
    if sp.ndim == 3:
        sp = sp[0]
    
    librosa.display.specshow(np.log(sp).T,
                            sr=sr,
                            hop_length=int(0.001 * sr * 5.0),
                            x_axis="time",
                            y_axis="linear",
                            cmap="magma")
    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
    plt.savefig(filename)  
    plt.close() 
    

    