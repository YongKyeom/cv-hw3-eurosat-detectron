"""
딥러닝 모델(Multi-Layer Perceptron / CNN 등)을 학습할 때 필요한 공통 유틸리티 함수 모음.

주요 기능:
    - 실행 디바이스 자동 선택(CUDA → MPS → CPU)
    - 랜덤 시드 고정(seed_everything)
    - 모델 파라미터 수 계산(count_parameters)
    - DataLoader worker 초기화(worker_init_fn)
    - 모델 요약 텍스트 생성(model_summary)
"""

from __future__ import annotations

import os
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.dl.cnn import CNNClassifier, CNNConfig
from model.dl.convnext import ConvNeXtClassifier, ConvNeXtConfig
from model.dl.mlp import MLPClassifier, MLPConfig
from model.dl.resnet import ResNetClassifier, ResNetConfig

# ---------------------------------------------------------------------------
# 🔵 Device 선택
# ---------------------------------------------------------------------------


def get_device() -> torch.device:
    """
    시스템에서 사용 가능한 device(CUDA → MPS → CPU)를 자동 선택한다.

    Returns:
        torch.device: cuda, mps, cpu 중 하나
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# 🔵 Random Seed 고정
# ---------------------------------------------------------------------------


def seed_everything(seed: int = 2025) -> None:
    """
    Python, NumPy, PyTorch의 랜덤 시드를 모두 고정한다.

    Args:
        seed (int): 시드 값

    Notes:
        reproducibility를 높이기 위해 cudnn deterministic 옵션을 활성화하지만,
        성능이 조금 느려질 수 있음.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# 🔵 DataLoader Worker Seed 설정
# ---------------------------------------------------------------------------


def worker_init_fn(worker_id: int) -> None:
    """
    DataLoader의 worker마다 서로 다른 시드를 적용한다.

    Args:
        worker_id (int): worker ID
    """
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(seed)


# ---------------------------------------------------------------------------
# 🔵 모델 파라미터 개수
# ---------------------------------------------------------------------------


def count_parameters(model: nn.Module) -> int:
    """
    학습 가능한 파라미터 수를 반환한다.

    Args:
        model (nn.Module): PyTorch 모델

    Returns:
        int: trainable parameters 개수
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# 🔵 모델 summary (torchsummary 없이 직접 구현)
# ---------------------------------------------------------------------------


def model_summary(model: nn.Module, input_size: Tuple[int, ...]) -> str:
    """
    torchsummary 없이 직접 구현한 간단한 모델 요약(summary).

    Args:
        model (nn.Module): PyTorch 모델
        input_size (Tuple[int, ...]): 입력 텐서 크기 (예: (3, 32, 32))

    Returns:
        str: summary 문자열
    """
    device = get_device()
    dummy = torch.zeros((1, *input_size)).to(device)

    summary_lines: List[str] = []
    summary_lines.append("=== Model Summary ===")

    def forward_hook(module, inp, out):
        class_name = module.__class__.__name__
        in_shape = tuple(inp[0].size())
        out_shape = tuple(out.size()) if isinstance(out, torch.Tensor) else "various"

        params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        summary_lines.append(f"{class_name:20s} | Input: {in_shape} -> Output: {out_shape} | Params: {params}")

    hooks = []
    for layer in model.modules():
        # 첫 번째 모듈(model 자체)은 skip
        if layer is model:
            continue
        hooks.append(layer.register_forward_hook(forward_hook))

    # Forward 실행
    _ = model(dummy)

    # hook 제거
    for h in hooks:
        h.remove()

    total_params = count_parameters(model)
    summary_lines.append(f"\nTotal Trainable Params: {total_params:,}")

    return "\n".join(summary_lines)


# ---------------------------------------------------------------------------
# 🔵 DataLoader 전체 라벨 수집
# ---------------------------------------------------------------------------


def collect_loader_targets(loader: DataLoader) -> np.ndarray:
    """DataLoader의 모든 배치 라벨을 numpy 배열로 이어 붙인다.

    Args:
        loader (DataLoader): 라벨을 수집할 PyTorch DataLoader.

    Returns:
        np.ndarray: 합쳐진 라벨 벡터(int64).
    """
    labels: List[int] = []
    for _, y in loader:
        labels.extend(np.asarray(y, dtype=np.int64).tolist())
    return np.asarray(labels, dtype=np.int64)


# ---------------------------------------------------------------------------
# 🔵 모델 빌더 (HW scripts 공유)
# ---------------------------------------------------------------------------

DEFAULT_DL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "mlp": {
        "hidden": [512, 256],
        "dropout": 0.3,
    },
    "cnn": {
        "channels": [32, 64, 128],
        "dropout": 0.3,
        "use_batchnorm": True,
    },
    "resnet": {
        "layers": (2, 2, 2),
        "base_channels": 32,
    },
    "convnext": {
        "depths": (2, 2, 2),
        "dims": (64, 128, 256),
    },
}


def build_dl_model(model_type: str, num_classes: int, params: Optional[Dict[str, Any]] = None) -> nn.Module:
    """EuroSAT 분류용 DL 모델을 생성한다.

    Args:
        model_type (str): {"mlp", "cnn", "resnet", "convnext"} 중 선택.
        num_classes (int): 분류 클래스 개수.
        params (Optional[Dict[str, Any]]): Hyperopt 결과 등으로부터 읽은 파라미터 dict.

    Returns:
        nn.Module: 생성된 PyTorch 분류 모델.

    Raises:
        ValueError: 지원하지 않는 model_type일 경우.
    """
    if model_type not in DEFAULT_DL_CONFIGS:
        raise ValueError(f"Unsupported DL model type: {model_type}")

    params = params or {}
    defaults = DEFAULT_DL_CONFIGS[model_type]

    # Set Random Seed
    seed_everything(2025)

    # Define model
    if model_type == "mlp":
        cfg = MLPConfig(
            input_dim=32 * 32 * 3,
            num_classes=num_classes,
            hidden_dims=params.get("hidden", defaults["hidden"]),
            dropout=float(params.get("dropout", defaults["dropout"])),
        )
        return MLPClassifier(cfg)

    if model_type == "cnn":
        cfg = CNNConfig(
            num_classes=num_classes,
            channels=params.get("channels", defaults["channels"]),
            dropout=float(params.get("dropout", defaults["dropout"])),
            use_batchnorm=bool(params.get("use_batchnorm", defaults["use_batchnorm"])),
        )
        return CNNClassifier(cfg)

    if model_type == "resnet":
        cfg = ResNetConfig(
            num_classes=num_classes,
            layers=tuple(params.get("layers", defaults["layers"])),
            base_channels=int(params.get("base_channels", defaults["base_channels"])),
        )
        return ResNetClassifier(cfg)

    cfg = ConvNeXtConfig(
        num_classes=num_classes,
        depths=tuple(params.get("depths", defaults["depths"])),
        dims=tuple(params.get("dims", defaults["dims"])),
    )
    return ConvNeXtClassifier(cfg)
