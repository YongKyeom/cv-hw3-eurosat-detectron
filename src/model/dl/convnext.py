"""
간단한 ConvNeXt 분류기 구현.

- ConvNeXt 구조를 32×32 입력에 맞게 간소화
- stage별 depth/dim은 config로 조정
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, layer_scale: float = 1e-6) -> None:
        super().__init__()
        # depthwise conv 후 Dw -> Norm -> 2x pointwise conv 구성
        # layer scaling → gamma로 구현
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)
        self.gamma = nn.Parameter(layer_scale * torch.ones(dim)) if layer_scale > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        shortcut = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma[:, None, None] * x
        return shortcut + x


@dataclass
class ConvNeXtConfig:
    num_classes: int = 10
    depths: Sequence[int] = (2, 2, 2)
    dims: Sequence[int] = (64, 128, 256)


class ConvNeXtClassifier(nn.Module):
    def __init__(self, config: ConvNeXtConfig) -> None:
        super().__init__()
        self.config = config

        self.downsample_layers = nn.ModuleList()
        # stem: 4x4 patchify → layer norm
        stem = nn.Sequential(
            nn.Conv2d(3, config.dims[0], kernel_size=4, stride=4),
            LayerNorm2d(config.dims[0]),
        )
        self.downsample_layers.append(stem)
        for i in range(len(config.dims) - 1):
            # 각 stage 시작 시 channel 변환 및 downsample
            down = nn.Sequential(
                LayerNorm2d(config.dims[i]),
                nn.Conv2d(config.dims[i], config.dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(down)

        stages = []
        for depth, dim in zip(config.depths, config.dims):
            # stage마다 설정된 depth만큼 ConvNeXtBlock 반복
            blocks = [ConvNeXtBlock(dim) for _ in range(depth)]
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)
        self.norm = nn.LayerNorm(config.dims[-1], eps=1e-6)
        self.head = nn.Linear(config.dims[-1], config.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Downsample + stage 반복 후 평균 pooling → Linear Head."""
        for down, stage in zip(self.downsample_layers, self.stages):
            x = down(x)
            x = stage(x)
        x = x.mean([-2, -1])
        x = self.norm(x)
        x = self.head(x)
        return x
