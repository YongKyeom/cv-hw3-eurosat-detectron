"""Detectron2 파인튜닝 설정 모음."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence


@dataclass(frozen=True)
class FineTuneSetting:
    """파인튜닝 하이퍼파라미터 한 세트.

    Attributes:
        label: 결과 폴더/로그 식별자.
        base_lr: Learning rate.
        max_iter: 학습 반복 횟수.
        ims_per_batch: 배치 크기.
    """

    label: str
    base_lr: float
    max_iter: int
    ims_per_batch: int = 2

    def to_dict(self) -> Dict[str, Any]:
        """dict 형태로 변환한다."""

        return {
            "label": self.label,
            "base_lr": self.base_lr,
            "max_iter": self.max_iter,
            "ims_per_batch": self.ims_per_batch,
        }


SWEEP_SETTINGS: Sequence[FineTuneSetting] = (
    FineTuneSetting("lr2.5e-4_iter400", base_lr=0.00025, max_iter=400, ims_per_batch=2),
    FineTuneSetting("lr1e-4_iter800", base_lr=0.0001, max_iter=800, ims_per_batch=2),
    FineTuneSetting("lr2.5e-4_iter600_batch4", base_lr=0.00025, max_iter=600, ims_per_batch=4),
)
