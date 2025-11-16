"""
EuroSAT Dataset Loader

- EuroSAT (RGB, 32x32) 분류를 위한 데이터셋/로더 생성 유틸리티.
- torchvision.datasets.ImageFolder 를 기반으로 동작.
- train/validation/test 자동 분할 기능 포함.
- augmentation 추가 가능.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import logging
import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

logger = logging.getLogger("hw03_logger")

# ---------------------------------------------------------------------------
# 🔵 유틸리티 — EuroSAT 다운로드 함수
# ---------------------------------------------------------------------------


def download_eurosat(root: Path) -> Path:
    """EuroSAT 압축 파일을 내려받고 압축을 푼다.

    Args:
        root (Path): 데이터셋 루트 디렉토리 (예: project_root/data).

    Returns:
        Path: 실제 이미지가 저장된 2750 폴더 경로.
    """
    import torchvision

    download_root = root / "EuroSAT"

    if (download_root / "2750").exists():
        logger.info("[dataset] EuroSAT already exists.")
        return download_root / "2750"

    logger.info("[dataset] Downloading EuroSAT...")
    download_root.mkdir(parents=True, exist_ok=True)

    torchvision.datasets.utils.download_and_extract_archive(
        url="http://madm.dfki.de/files/sentinel/EuroSAT.zip",
        download_root=str(download_root),
        md5="c8fa014336c82ac7804f0398fcb19387",
        remove_finished=True,
    )

    logger.info("[dataset] Download complete.")
    return download_root / "2750"


# ---------------------------------------------------------------------------
# 🔵 Dataset Config
# ---------------------------------------------------------------------------


@dataclass
class EuroSATConfig:
    """
    EuroSAT 데이터셋 구성 옵션.

    Attributes:
        root (Path): EuroSAT 데이터를 저장할 루트
        img_size (int): 이미지 크기 (H=W)
        train_ratio (float): train 비율
        val_ratio (float): validation 비율
        test_ratio (float): test 비율
        batch_size (int): DataLoader 배치 크기
        num_workers (int): DataLoader worker 수
        augment (bool): augmentation 사용 여부
    """

    root: Path
    img_size: int = 32

    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    batch_size: int = 64
    num_workers: int = 2

    augment: bool = False


# ---------------------------------------------------------------------------
# 🔵 Transform 생성 함수
# ---------------------------------------------------------------------------


def build_transforms(img_size: int, augment: bool = False) -> Callable:
    """EuroSAT 이미지 전처리 파이프라인을 생성한다.

    Args:
        img_size (int): 최종 이미지 크기(가로/세로 동일).
        augment (bool): True면 랜덤 플립/회전/색상변조를 추가한다.

    Returns:
        Callable: torchvision transform.
    """
    tf_list: List[Callable] = [
        transforms.Resize((img_size, img_size)),
    ]

    if augment:
        tf_list.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=25),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.05,
                ),
            ]
        )

    tf_list.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            ),
        ]
    )

    return transforms.Compose(tf_list)


# ---------------------------------------------------------------------------
# 🔵 Main loader function
# ---------------------------------------------------------------------------


class _TransformedSubset(Dataset):
    """Subset 에 transform 을 후처리로 적용하는 래퍼."""

    def __init__(self, subset: Subset, transform: Optional[Callable]):
        """Subset 과 transform 을 저장한다.

        Args:
            subset (Subset): 원본 Subset 객체.
            transform (Optional[Callable]): 적용할 torchvision transform.
        """
        self.subset = subset
        self.transform = transform

    def __len__(self) -> int:
        """Subset 길이를 반환한다.

        Returns:
            int: 샘플 개수.
        """
        return len(self.subset)

    def __getitem__(self, idx: int):
        """원본 이미지를 가져온 뒤 transform 을 적용한다.

        Args:
            idx (int): 가져올 인덱스.

        Returns:
            Tuple[Any, int]: 변환된 이미지와 라벨.
        """
        img, label = self.subset[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def load_eurosat(
    config: EuroSATConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader, List[str]]:
    """EuroSAT을 train/val/test로 분할하고 DataLoader를 생성한다.

    Args:
        config (EuroSATConfig): 데이터 구성 및 로더 파라미터.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader, DataLoader, List[str]]: (train, train_eval, val,
            test) DataLoader와 클래스 이름 리스트.
    """
    data_root = download_eurosat(config.root)

    train_transform = build_transforms(config.img_size, config.augment)
    eval_transform = build_transforms(config.img_size, augment=False)

    # ImageFolder 기반 Dataset (transform=None, Subset 단계에서 적용)
    dataset = ImageFolder(root=str(data_root), transform=None)
    class_names = dataset.classes

    # 데이터 분할 --------------------------------------------------------------
    n = len(dataset)
    indices = np.arange(n)
    np.random.shuffle(indices)

    n_train = int(n * config.train_ratio)
    n_val = int(n * config.val_ratio)
    # n_test = n - n_train - n_val

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    base_train = Subset(dataset, train_idx)
    base_val = Subset(dataset, val_idx)
    base_test = Subset(dataset, test_idx)

    train_ds = _TransformedSubset(base_train, train_transform)
    train_eval_ds = _TransformedSubset(base_train, eval_transform)
    val_ds = _TransformedSubset(base_val, eval_transform)
    test_ds = _TransformedSubset(base_test, eval_transform)

    # DataLoader 생성 ---------------------------------------------------------
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )

    train_eval_loader = DataLoader(
        train_eval_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    logger.info(
        "[dataset] Train: %d | Val: %d | Test: %d",
        len(train_ds),
        len(val_ds),
        len(test_ds),
    )

    return train_loader, train_eval_loader, val_loader, test_loader, class_names
