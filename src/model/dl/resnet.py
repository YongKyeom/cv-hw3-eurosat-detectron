"""
간단한 ResNet 분류기.

- 32×32 RGB 이미지를 위한 소형 ResNet-18 변형모델
- ResNet-18 구조를 간소화한 3-stage 네트워크
- config로 block 수와 channel수를 조절
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn


def _conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    """3x3 Conv + padding=1 조합을 단순화한 헬퍼."""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


class ResNetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        # 첫 번째 conv: stride 조절을 통해 spatial downsampling 제어
        self.conv1 = _conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # 두 번째 conv: 출력 channel을 유지하면서 residual 연결
        self.conv2 = _conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            # 입력과 출력의 크기가 다를 경우 residual path를 1x1 conv로 정렬
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # Residual connection
        out += identity

        out = self.relu(out)
        return out


@dataclass
class ResNetConfig:
    num_classes: int = 10
    layers: Sequence[int] = (2, 2, 2)
    base_channels: int = 32


class ResNetClassifier(nn.Module):
    def __init__(self, config: ResNetConfig) -> None:
        super().__init__()
        self.config = config

        self.in_channels = config.base_channels
        # stem conv는 3x3, stride=1로 두어 작은 입력에 적합하도록 조정
        self.conv1 = nn.Conv2d(3, config.base_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(config.base_channels)
        self.relu = nn.ReLU(inplace=True)

        channels = config.base_channels
        # layer1: stride=1, 이후 stage마다 채널을 2배씩 증가시키며 stride=2 downsample
        self.layer1 = self._make_layer(channels, config.layers[0], stride=1)
        self.layer2 = self._make_layer(channels * 2, config.layers[1], stride=2)
        self.layer3 = self._make_layer(channels * 4, config.layers[2], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels * 4, config.num_classes)

    def _make_layer(self, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        """block 수와 stride 정보를 바탕으로 stage를 구성한다."""
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for s in strides:
            layers.append(ResNetBlock(self.in_channels, out_channels, stride=s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """입력을 stage별로 처리하여 class logits을 반환한다."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
