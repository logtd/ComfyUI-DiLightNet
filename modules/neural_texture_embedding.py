from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

import comfy.ops
ops = comfy.ops.disable_weight_init


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(
            ops.Conv2d(dim, dim, 3, 1, 1),
            ops.GroupNorm(num_groups=8, num_channels=dim),
            nn.SiLU(inplace=True),
            ops.Conv2d(dim, dim, 3, 1, 1),
        )

    def forward(self, x):
        return x + self.conv(x)


class NeuralTextureEncoder(nn.Module):
    def __init__(self, in_dim=3, out_dim=16, dims=(32, 64, 128), groups=8):
        super().__init__()
        self.model = nn.Sequential(
            ops.Conv2d(in_dim, dims[0], kernel_size=3, padding=1),
            nn.SiLU(inplace=True),

            # down 1
            ops.Conv2d(dims[0], dims[1], kernel_size=3, padding=1, stride=2),
            ops.GroupNorm(num_groups=groups, num_channels=dims[1]),
            nn.SiLU(inplace=True),

            # down 2
            ops.Conv2d(dims[1], dims[2], kernel_size=3, padding=1, stride=2),
            ops.GroupNorm(num_groups=groups, num_channels=dims[2]),
            nn.SiLU(inplace=True),

            # res blocks
            ResBlock(dims[2]),
            ResBlock(dims[2]),
            ResBlock(dims[2]),
            ResBlock(dims[2]),

            # up 1
            ops.ConvTranspose2d(dims[2], dims[1], kernel_size=4, padding=1, stride=2),
            ops.GroupNorm(num_groups=groups, num_channels=dims[1]),
            nn.SiLU(inplace=True),

            # up 2
            ops.ConvTranspose2d(dims[1], dims[0], kernel_size=4, padding=1, stride=2),
            ops.GroupNorm(num_groups=groups, num_channels=dims[0]),
            nn.SiLU(inplace=True),

            # out
            ops.Conv2d(dims[0], out_dim, kernel_size=3, padding=1),
        )
        self.gradient_checkpointing = False

    def forward(self, x):
        x = self.model(x)
        return x


class NeuralTextureEmbedding(nn.Module):
    def __init__(
            self,
            conditioning_embedding_channels: int,
            conditioning_channels: int = 3,
            block_out_channels: Tuple[int] = (16, 32, 96, 256),
            shading_hint_channels: int = 12,  # diffuse + 3 * ggx
    ):
        super().__init__()
        self.conditioning_channels = conditioning_channels
        self.shading_hint_channels = shading_hint_channels

        self.conv_in = ops.Conv2d(shading_hint_channels, block_out_channels[0], kernel_size=3, padding=1)
        self.neural_texture_encoder = NeuralTextureEncoder(in_dim=conditioning_channels, out_dim=shading_hint_channels)

        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(ops.Conv2d(channel_in, channel_in, kernel_size=3, padding=1))
            self.blocks.append(ops.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=2))

        self.conv_out = ops.Conv2d(
            block_out_channels[-1],
            conditioning_embedding_channels,
            kernel_size=3,
            padding=1
        )

    def forward(self, all_conditioning, emb=None, context=None):
        # conditioning: [BS, 4 + 12, 512, 512]  # RGB ref image + shading hint (diffuse + 3 * ggx)
        conditioning, shading_hint = torch.split(
            all_conditioning,
            [self.conditioning_channels, self.shading_hint_channels],
            dim=1
        )
        embedding = self.neural_texture_encoder(conditioning)  # [BS, 15, 512, 512]

        # multiply shading hint to each channel
        embedding = embedding * shading_hint
        embedding = self.conv_in(embedding)
        embedding = F.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)

        embedding = self.conv_out(embedding)

        return embedding
