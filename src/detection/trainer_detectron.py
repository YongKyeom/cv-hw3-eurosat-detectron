"""
Detectron2 Trainer 모듈
    - DefaultTrainer 기반 Fine-tuning
    - Balloon dataset 전용 Trainer
"""

from __future__ import annotations

import os

from detectron2.data import build_detection_train_loader
from detectron2.engine import DefaultTrainer
from detectron2.utils.events import CommonMetricPrinter, EventWriter, JSONWriter


class BalloonTrainer(DefaultTrainer):
    """
    Detectron2 DefaultTrainer 확장 버전.
    필요하면 augmentation 등을 추가한 custom dataloader 구성 가능.
    """

    def build_writers(self) -> list[EventWriter]:
        """학습 중 로그를 기록할 writer 리스트를 생성한다.
        TensorBoard 오류로 TensorBoard 의존성을 제거하고 콘솔 로그와 JSON 로그만 사용하도록 한다.

        - CommonMetricPrinter: 학습 진행 상황을 콘솔에 사람이 읽기 쉬운 형태로 출력
        - JSONWriter: 학습 지표를 JSON 파일로 저장 (cfg.OUTPUT_DIR/metrics.json)
        """
        cfg = self.cfg
        return [
            CommonMetricPrinter(max_iter=cfg.SOLVER.MAX_ITER),
            JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
        ]

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
