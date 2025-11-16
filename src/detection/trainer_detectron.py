"""
Detectron2 Trainer 모듈
    - DefaultTrainer 기반 Fine-tuning
    - Balloon dataset 전용 Trainer
"""

from __future__ import annotations

import os

from detectron2.data import build_detection_train_loader
from detectron2.engine import DefaultTrainer


class BalloonTrainer(DefaultTrainer):
    """
    Detectron2 DefaultTrainer 확장 버전.
    필요하면 augmentation 등을 추가한 custom dataloader 구성 가능.
    """

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Data augmentation 적용하려면 이 메서드에서 커스터마이징 가능.
        기본은 Detectron2의 standard loader 사용.
        """
        return build_detection_train_loader(cfg)

    def train_with_output(self):
        """
        train()을 실행하고, cfg.OUTPUT_DIR 경로가 존재하도록 보장.
        """
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
        self.resume_or_load(resume=False)
        return self.train()
