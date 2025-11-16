"""
CNN Classifier for EuroSAT or general 32×32 RGB image classification.

- 3× Conv + BatchNorm + ReLU + MaxPool 구조
- Fully Connected classifier block
- hidden_dims, dropout 비율 등을 설정으로 조정 가능
- TrainerTorch(trainer_torch.py)에서 학습 가능한 구조
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# 🔵 CNN 설정값
# ---------------------------------------------------------------------------


@dataclass
class CNNConfig:
    """
    CNN 모델의 하이퍼파라미터 구성.

    Attributes:
        num_classes (int): 최종 분류 클래스 수.
        channels (List[int]): 각 Conv 블록의 output channel 수.
        dropout (float): fc layer dropout 비율.
        use_batchnorm (bool): BatchNorm 사용 여부.
    """

    num_classes: int = 10
    channels: List[int] = None
    dropout: float = 0.3
    use_batchnorm: bool = True


# ---------------------------------------------------------------------------
# 🔵 CNN Classifier
# ---------------------------------------------------------------------------


class CNNClassifier(nn.Module):
    """
    32×32 RGB 입력용 CNN 기반 분류기.

    구조:
        Conv1 → BN → ReLU → MaxPool
        Conv2 → BN → ReLU → MaxPool
        Conv3 → BN → ReLU → MaxPool
        → Flatten → FC → ReLU → Dropout → FC(num_classes)

    Conv 설정은 CNNConfig.channels 로 조정 (예: [32, 64, 128])
    """

    def __init__(self, config: Optional[CNNConfig] = None):
        super().__init__()

        if config is None:
            config = CNNConfig(channels=[32, 64, 128])

        self.config = config

        chs = config.channels
        if chs is None or len(chs) < 1:
            raise ValueError("CNNConfig.channels 는 최소 1개 이상의 채널 수를 포함해야 합니다.")

        layers: List[nn.Module] = []
        in_c = 3  # 입력 이미지 RGB 3채널

        # Conv Blocks 구성 -------------------------------------------------
        for out_c in chs:
            layers.append(nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1))
            if config.use_batchnorm:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_c = out_c

        self.features = nn.Sequential(*layers)

        # 최종 feature map 크기 계산 ----------------------------------------
        # Conv 구성을 변경해도 자동으로 flatten dimension을 계산하도록 더미 입력을 통과시킨다.
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 32, 32)
            fm = self.features(dummy)
            fm_size = int(fm.shape[1] * fm.shape[2] * fm.shape[3])

        # Classifier block ---------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Linear(fm_size, 256),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, config.num_classes),
        )

    # ----------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward 계산.

        Args:
            x (Tensor): (B, 3, 32, 32) 입력 이미지

        Returns:
            Tensor: (B, num_classes) 로짓(logit)
        """
        out = self.features(x)  # (B, C3, 4, 4)
        out = torch.flatten(out, 1)  # (B, C3*4*4)
        out = self.classifier(out)  # (B, num_classes)
        return out

    # ----------------------------------------------------------------------
    def count_parameters(self) -> int:
        """
        학습 가능한 파라미터 수 반환.

        Returns:
            int: 파라미터 개수
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
