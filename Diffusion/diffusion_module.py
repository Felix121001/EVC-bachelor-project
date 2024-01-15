import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import glob


from einops import rearrange
from functools import partial
from inspect import isfunction
from einops import rearrange, reduce, repeat
from torch import einsum

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
import torch.optim as optim
from math import log, sqrt
import math


def exists(x):
    return x is not None


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, max_positions=10000):
        super().__init__()
        self.dim = dim
        self.max_positions = max_positions

    def forward(self, x):
        device = x.device  # Get the device from the input tensor
        x = x.float()
        half_dim = self.dim // 2
        emb = math.log(self.max_positions) / (half_dim - 1)
        emb = torch.exp(
            torch.arange(half_dim, device=device, dtype=torch.float32) * -emb
        )
        emb = x[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class Identity(nn.Module):
    def forward(self, x):
        return x


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


def Upsample(dim):
    return nn.ConvTranspose2d(dim, dim, kernel_size=4, stride=2, padding=1)


def Downsample(dim):
    return nn.Conv2d(dim, dim, kernel_size=4, stride=2, padding=1)


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.norm = nn.LayerNorm([dim], eps=eps)

    def forward(self, x):
        n, c, h, w = x.size()
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        return x.permute(0, 3, 1, 2).contiguous()


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class GELU(nn.Module):
    def __init__(self, approximate=False):
        super().__init__()
        self.approximate = approximate

    def forward(self, x):
        if self.approximate:
            return (
                x
                * 0.5
                * (
                    1.0
                    + torch.tanh(sqrt(2 / 3.1415) * (x + 0.044715 * torch.pow(x, 3)))
                )
            )
        else:
            return x * 0.5 * (1.0 + torch.erf(x / sqrt(2.0)))


class Block(nn.Module):
    def __init__(self, dim_in, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv2d(dim_in, dim_out, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out, eps=1e-05)
        self.act = nn.SiLU()

    def forward(self, x, gamma_beta=None):
        x = self.proj(x)
        x = self.norm(x)

        if gamma_beta is not None:
            gamma, beta = gamma_beta
            gamma = gamma.unsqueeze(-1).unsqueeze(-1)
            beta = beta.unsqueeze(-1).unsqueeze(-1)
            x = x * (gamma + 1) + beta

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if time_emb_dim is not None
            else None
        )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = (
            nn.Conv2d(dim, dim_out, kernel_size=1) if dim != dim_out else nn.Identity()
        )

    def forward(self, x, time_emb=None):
        gamma_beta = None
        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb)
            gamma_beta = torch.chunk(time_emb, 2, dim=-1)

        h = self.block1(x, gamma_beta=gamma_beta)
        h = self.block2(h)
        return h + self.res_conv(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.to_out = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1), LayerNorm(dim))

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h d) x y -> b h x y d", h=self.heads), qkv
        )

        q = q.softmax(dim=-1)
        k = k.softmax(dim=-2)

        context = einsum("b h x y d, b h x y e -> b h d e", k, v)
        out = einsum("b h d e, b h x y e -> b h x y d", context, q)
        out = rearrange(out, "b h x y d -> b (h d) x y")
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.to_out = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h d) x y -> b h x y d", h=self.heads), qkv
        )

        q = q * self.scale
        sim = einsum("b h x y d, b h x y e -> b h y d e", q, k)
        attn = sim.softmax(dim=-1)

        out = einsum("b h x d e, b h x y e -> b h x y d", attn, v)
        out = rearrange(out, "b h x y d -> b (h d) x y")
        return self.to_out(out)


class MLP(nn.Module):
    def __init__(self, hidden_dim):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(start_dim=-1),  # Flattening the input tensor
            nn.Linear(1, hidden_dim),  # Dense layer
            nn.GELU(),  # GELU Activation
            nn.LayerNorm(hidden_dim),  # Layer Normalization
            nn.Linear(hidden_dim, hidden_dim),  # Dense layer
            nn.GELU(),  # GELU Activation
            nn.LayerNorm(hidden_dim),  # Layer Normalization
            nn.Linear(hidden_dim, hidden_dim),  # Dense layer
        )

    def forward(self, x):
        return self.net(x)


class ClassConditioning(nn.Module):
    def __init__(self, height, width, inp, num_channels=1):
        super(ClassConditioning, self).__init__()

        self.num_channels = num_channels
        self.height = height
        self.width = width
        self.linear = nn.Linear(
            in_features=inp, out_features=height * width * num_channels
        )

        self.activation = nn.SiLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        x = x.view(
            -1, self.num_channels, self.height, self.width
        )  # Assuming input x is of shape (batch_size, inp)
        return x


class Unet_conditional(nn.Module):
    def __init__(
        self,
        dim=64,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        resnet_block_groups=8,
        learned_variance=False,
        sinusoidal_cond_mlp=True,
        num_classes=None,
        class_embedder=None,
        class_emb_dim=32,
        height=32,
        width=32,
        device=None,
    ):
        super(Unet_conditional, self).__init__()

        self.device = device

        self.channels = channels
        self.height = height
        self.width = width
        self.class_emb_dim = class_emb_dim

        # print("Unet_conditional height: ", height)
        # print("Unet_conditional width: ", width)

        self.class_embeddings = (
            nn.Embedding(num_classes, class_emb_dim)
            if class_embedder is None
            else class_embedder
        )

        init_dim = init_dim or (dim // 3 * 2)
        # print(f"Init Dim: {init_dim}")
        self.init_conv = nn.Conv2d(
            in_channels=channels,
            out_channels=init_dim,
            kernel_size=7,
            stride=1,
            padding=3,
        )

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print("in_out: ", in_out)

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        time_dim = dim * 4
        self.sinusoidal_cond_mlp = sinusoidal_cond_mlp

        if sinusoidal_cond_mlp:
            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(dim),
                nn.Linear(dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )
        else:
            self.time_mlp = MLP(time_dim)

        self.downs = nn.ModuleList()  # A list to hold the downsampling blocks
        self.ups = nn.ModuleList()  # A list to hold the upsampling blocks

        now_height = height
        now_width = width

        # Defining downsample blocks
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_block = nn.ModuleList(
                [
                    ClassConditioning(now_height, now_width, self.class_emb_dim),
                    block_klass(dim_in + 1, dim_out, time_emb_dim=time_dim),
                    block_klass(dim_out, dim_out, time_emb_dim=time_dim),
                    Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                    Downsample(dim_out) if not is_last else nn.Identity(),
                ]
            )

            self.downs.append(down_block)
            now_height //= 2 if not is_last else 1
            now_width //= 2 if not is_last else 1

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 2)
            up_block = nn.ModuleList(
                [
                    ClassConditioning(now_height, now_width, self.class_emb_dim),
                    block_klass((dim_out * 2) + 1, dim_in, time_emb_dim=time_dim),
                    block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                    Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                    Upsample(dim_in),  # if not is_last else nn.Identity()
                ]
            )

            self.ups.append(up_block)
            now_height *= 2 if not is_last else 1
            now_width *= 2 if not is_last else 1

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = out_dim if out_dim is not None else default_out_dim

        self.final_conv = nn.Sequential(
            block_klass(dim * 2, dim),
            nn.Conv2d(dim, self.out_dim, kernel_size=1, stride=1),
        )

    def forward(self, x, time=None, class_vector=None, training=True):
        # print(f"Input Shape: {x.shape}") #torch.Size([64, 1, 32, 32])
        x = self.init_conv(x)
        # print(f"After init_conv Shape: {x.shape}") #torch.Size([64, 42, 32, 32])
        t = self.time_mlp(time)

        # print(f"Class Vector Shape: {class_vector.shape}") #torch.Size([64])
        class_vector = self.class_embeddings(class_vector.int())
        # print(f"Class Vector Shape: {class_vector.shape}") #torch.Size([64, 32])

        h = []

        for it, down_block in enumerate(self.downs):
            class_conditioning, block1, block2, attn, downsample = down_block
            # print("unsqueeze shape: ", class_vector.unsqueeze(1).shape)
            cv = class_conditioning(class_vector.unsqueeze(1))
            #print(f"Input Shape in for: {x.shape}") #torch.Size([64, 42, 32, 32])
            #print(f"After Class Conditioning Shape: {cv.shape}") #torch.Size([64, 1, 32, 32])
            x = torch.cat([x, cv], dim=1)  # torch.Size([64, 43, 32, 32])
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)
            # print("Downsampled Shape: ", x.shape)
        for up_block in self.ups:
            class_conditioning, block1, block2, attn, upsample = up_block
            cv = class_conditioning(class_vector)
            #print("Class Conditioning Shape: ", cv.shape)
            #print("x Shape: ", x.shape)
            x = torch.cat([x, cv], dim=1)
            #print("Concatenated Shape: ", x.shape)
            #print("h Shape: ", h[-1].shape)
            x = torch.cat([x, h.pop()], dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)
            # print("Upsampled Shape: ", x.shape)

        x = torch.cat([x, h.pop()], dim=1)
        x = self.final_conv(x)

        return x
