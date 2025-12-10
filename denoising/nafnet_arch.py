"""
Copyright (c) 2025 Samsung Electronics Co., Ltd.

Author(s):
Raghav Goyal (raghav.goyal@samsung.com)

NAFNet architecture
This code is modified from the original NAFNet repository and follows the same
license as the original repo (MIT License).
"""

import torch
import torch.nn as nn


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, in_channels, dw_expand=2, ffn_expand=2):
        super().__init__()
        dw_channels = in_channels * dw_expand
        self.conv1 = nn.Conv2d(in_channels, dw_channels, kernel_size=(1, 1))
        self.conv2 = nn.Conv2d(dw_channels, dw_channels, kernel_size=(3, 3), padding=(1, 1), groups=dw_channels)
        self.conv3 = nn.Conv2d(dw_channels // 2, in_channels, kernel_size=(1, 1))

        # channel attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channels // 2, dw_channels // 2, kernel_size=(1, 1))
        )

        # simple gate
        self.sg = SimpleGate()

        ffn_channels = ffn_expand * in_channels
        self.conv4 = nn.Conv2d(in_channels, ffn_channels, kernel_size=(1, 1))
        self.conv5 = nn.Conv2d(ffn_channels // 2, in_channels, kernel_size=(1, 1))

        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)

        self.beta = nn.Parameter(torch.zeros((1, in_channels, 1, 1)))
        self.gamma = nn.Parameter(torch.zeros((1, in_channels, 1, 1)))

    def forward(self, img):
        x = img

        # layer norm 2d
        x = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        y = img + x * self.beta

        # layer norm 2d
        y_norm = self.norm2(y.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.conv4(y_norm)
        x = self.sg(x)
        x = self.conv5(x)

        return y + x * self.gamma


class NAFNet(nn.Module):
    def __init__(self, width, encoder_block_nums, middle_block_num, decoder_block_nums):
        super().__init__()

        self.in_layer = nn.Conv2d(3, width, kernel_size=(3, 3), padding=(1, 1))
        self.out_layer = nn.Conv2d(width, 3, kernel_size=(3, 3), padding=(1, 1))

        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.middle_blocks = nn.ModuleList()
        self.up_layers = nn.ModuleList()
        self.down_layers = nn.ModuleList()

        # init channel
        channel_count = width

        # encoder
        for encoder_num in encoder_block_nums:
            self.encoder_blocks.append(
                nn.Sequential(*[NAFBlock(channel_count) for _ in range(encoder_num)])
            )
            self.down_layers.append(
                nn.Conv2d(channel_count, 2 * channel_count, kernel_size=(2, 2), stride=(2, 2))
            )
            channel_count = channel_count * 2

        # middle
        self.middle_blocks = nn.Sequential(
            *[NAFBlock(channel_count) for _ in range(middle_block_num)])

        # decoder
        for decoder_num in decoder_block_nums:
            self.up_layers.append(nn.Sequential(
                nn.Conv2d(channel_count, channel_count * 2, kernel_size=(1, 1), bias=False),
                nn.PixelShuffle(2)
            ))
            channel_count = channel_count // 2
            self.decoder_blocks.append(
                nn.Sequential(*[NAFBlock(channel_count) for _ in range(decoder_num)]))

        self.pad_size = 2 ** len(self.encoder_blocks)

    def forward(self, img):
        H_wo_pad, W_wo_pad = img.shape[2:]
        img = self.pad_image_wrt_size(img)

        x = self.in_layer(img)

        # encoder
        encoder_intermediates = []
        for _encoder_block, _down_layer in zip(self.encoder_blocks, self.down_layers):
            x = _encoder_block(x)
            encoder_intermediates.append(x)
            x = _down_layer(x)

        # middle
        x = self.middle_blocks(x)

        # decoder
        for _decoder_block, _up_layer, _encoder_skip in zip(self.decoder_blocks, self.up_layers, encoder_intermediates[::-1]):
            x = _up_layer(x)
            x = x + _encoder_skip
            x = _decoder_block(x)

        x = self.out_layer(x)
        x = x + img

        return x[:, :, :H_wo_pad, :W_wo_pad]

    def pad_image_wrt_size(self, img):
        H, W = img.shape[2:]
        pad_h = (self.pad_size - H % self.pad_size) % self.pad_size
        pad_w = (self.pad_size - W % self.pad_size) % self.pad_size
        img = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h))
        return img
