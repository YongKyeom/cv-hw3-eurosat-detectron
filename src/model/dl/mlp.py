"""
MLP(Classifier) for EuroSAT or general image classification.

- 입력 이미지를 32×32 RGB 기준으로 flatten 후 fully-connected MLP로 분류.
- hidden_dims, dropout, activation 등을 파라미터로 선택 가능.
- TrainerTorch(trainer_torch.py)에서 직접 학습할 수 있도록 설계됨.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# 🔵 Activation Factory
# ---------------------------------------------------------------------------


def get_activation(name: str) -> nn.Module:
    """
    문자열 기반으로 활성화 함수를 반환한다.

    Args:
        name (str): {"relu", "gelu", "tanh", "sigmoid"}

    Returns:
        nn.Module: 활성화 함수
    """
    name = name.lower()

    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()

    raise ValueError(f"지원하지 않는 activation 함수: {name}")


# ---------------------------------------------------------------------------
# 🔵 Multi-Layer Perceptron (for 32x32 RGB)
# ---------------------------------------------------------------------------


@dataclass
class MLPConfig:
    """
    MLP 모델 구성 설정 값.

    Attributes:
        input_dim (int): 입력 벡터 차원 (기본: 32*32*3)
        num_classes (int): 분류 클래스 개수
        hidden_dims (List[int]): 히든 레이어 크기 리스트
        activation (str): 활성화 함수 종류
        dropout (float): 드롭아웃 비율
    """

    input_dim: int = 32 * 32 * 3
    num_classes: int = 10
    hidden_dims: List[int] = None
    activation: str = "relu"
    dropout: float = 0.2


class MLPClassifier(nn.Module):
    """
    다층 퍼셉트론 기반 이미지 분류기.

    - forward는 (B, input_dim) → (B, num_classes) 출력.
    - hidden_dims 길이에 따라 레이어 깊이 자동 구성.
    - TrainerTorch가 cross-entropy loss로 학습하도록 설계됨.
    """

    def __init__(self, config: Optional[MLPConfig] = None) -> None:
        super().__init__()

        if config is None:
            config = MLPConfig(hidden_dims=[512, 256])

        self.config = config

        # 활성화 함수 생성
        act = get_activation(config.activation)

        layers: List[nn.Module] = []
        in_dim = config.input_dim

        # hidden layer 구성
        if config.hidden_dims is None or len(config.hidden_dims) == 0:
            raise ValueError("hidden_dims는 최소 1개 이상 지정해야 합니다.")

        for h in config.hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(act)
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            in_dim = h

        # classifier layer
        layers.append(nn.Linear(in_dim, config.num_classes))

        self.net = nn.Sequential(*layers)

    # ----------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward 계산.

        Args:
            x (Tensor): 입력 이미지 텐서 (B, 3, 32, 32)

        Returns:
            Tensor: (B, num_classes) 로짓(logit)
        """
        # 이미지 → flatten
        x = x.view(x.size(0), -1)
        return self.net(x)

    # ----------------------------------------------------------------------
    def count_parameters(self) -> int:
        """
        학습 가능한 파라미터 개수 반환.

        Returns:
            int: 파라미터 수
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
