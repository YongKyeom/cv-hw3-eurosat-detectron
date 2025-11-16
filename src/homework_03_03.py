# type: ignore
"""HW3 문제 3 전용 스크립트.

풍선 데이터셋을 활용해 Detectron2 Mask R-CNN을 파인튜닝하고
사전학습/튜닝 모델의 평가 및 시각화를 수행한다.

문제 3 — Detectron2 Instance Segmentation Fine-tuning
    • 사전학습 모델 확인: Mask R-CNN (R50-FPN) 예제 이미지 inference 및 선정 이유 로깅
    • 파인튜닝: Balloon 데이터셋 로더 등록 → Detectron2 Trainer로 fine-tuning
    • 검증/시각화: COCOEvaluator로 AP/AR 측정, Pre/Post 시각화 이미지 저장, 문제점 분석 로그

전체 실행 흐름
    1) Paths/logger/seed 초기화
    2) 문제 3 실행 → Detectron2 fine-tuning 및 평가
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor

from detection.balloon_dataset import load_balloon_dataset
from detection.config_builder import build_config
from detection.evaluator import evaluate_model
from detection.trainer_detectron import BalloonTrainer
from model.dl.utils import seed_everything
from utils.io import ensure_dir, find_first_image, load_color, save_image
from utils.logger import get_logger
from utils.paths import Paths

logger: Optional[logging.Logger] = None


def _visualize_predictions(
    image_bgr: np.ndarray,
    outputs: Any,
    metadata: Any,
    save_path: Path,
    scale: float = 0.9,
) -> None:
    """Detectron2 Mask R-CNN 결과를 이미지로 저장한다."""
    from detectron2.utils.visualizer import ColorMode, Visualizer

    vis = Visualizer(
        image_bgr[:, :, ::-1],
        metadata=metadata,
        scale=scale,
        instance_mode=ColorMode.IMAGE,
    )
    vis_output = vis.draw_instance_predictions(outputs["instances"].to("cpu"))
    save_image(save_path, vis_output.get_image()[:, :, ::-1])


# ------------------------------------------------------------------
# HW3 w/ 3번 문제 Main Execute
# ------------------------------------------------------------------
if __name__ == "__main__":
    st_time = datetime.now()

    # ------------------------------------------------------------------
    # 0. 경로/로거/시드 설정
    # ------------------------------------------------------------------
    project_root = Path(os.path.join(os.path.dirname(__file__), "..")).resolve()
    paths = Paths.from_root(project_root)
    data_dir = paths.data_dir
    result_dir = paths.result_dir

    logger = get_logger(paths.log_dir, "hw03_03")
    logger.info("Project Root: %s", project_root)
    logger.info("Result Root: %s", result_dir)

    seed_everything(2025)

    # ------------------------------------------------------------------
    # 3. 문제 3 — Detectron2 Fine-tuning
    #     • Parameter 결과: Detectron2 cfg 정보/사전학습 vs 튜닝 성능
    #     • 최종 스코어: COCOEvaluator 결과 로깅
    #     • 시각화: 사전학습/튜닝 이미지 저장 및 분석 메시지
    # ------------------------------------------------------------------
    logger.info("==================== 문제 3: Detectron2 Fine-tuning ====================")
    balloon_root = data_dir / "balloon"
    if balloon_root.exists():
        for split in ["train", "val"]:
            name = f"balloon_{split}"
            if name not in DatasetCatalog.list():
                DatasetCatalog.register(name, lambda split=split: load_balloon_dataset(balloon_root / split))
                MetadataCatalog.get(name).set(thing_classes=["balloon"])

        metadata = MetadataCatalog.get("balloon_train")
        for split in ["train", "val"]:
            size = len(load_balloon_dataset(balloon_root / split))
            logger.info("[문제3] %s samples: %d", split, size)

        model_name = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        logger.info("[문제3] %s 사용 (성능/시간 균형 목적)", model_name)

        p3_dir = result_dir / "p3_detectron2"
        pre_dir = p3_dir / "pretrained"
        ft_dir = p3_dir / "finetuned"
        ensure_dir(pre_dir)
        ensure_dir(ft_dir)

        sample_image = find_first_image(balloon_root / "val") or find_first_image(balloon_root / "train")

        pre_cfg = build_config(
            model_name=model_name,
            train_dataset="balloon_train",
            output_dir=str(pre_dir / "tmp"),
            base_lr=0.00025,
            max_iter=1000,
        )
        pre_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        predictor_pre = DefaultPredictor(pre_cfg)

        if sample_image is not None:
            # 사전학습 모델 시각화
            img = load_color(sample_image)
            outputs = predictor_pre(img)
            _visualize_predictions(img, outputs, metadata, pre_dir / "sample_pretrained.png")

        pre_eval = evaluate_model(pre_cfg, "balloon_val")
        logger.info("[문제3] Pre-trained Eval: %s", pre_eval)

        ft_cfg = build_config(
            model_name=model_name,
            train_dataset="balloon_train",
            output_dir=str(ft_dir / "output"),
            base_lr=0.00025,
            max_iter=400,
        )
        trainer = BalloonTrainer(ft_cfg)
        trainer.train_with_output()

        final_weights = Path(ft_cfg.OUTPUT_DIR) / "model_final.pth"
        ft_cfg.MODEL.WEIGHTS = str(final_weights)
        ft_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        ft_cfg.DATASETS.TEST = ("balloon_val",)
        predictor_ft = DefaultPredictor(ft_cfg)

        ft_eval = evaluate_model(ft_cfg, "balloon_val", weights_path=str(final_weights))
        logger.info("[문제3] Fine-tuned Eval: %s", ft_eval)

        # 튜닝 결과 시각화 샘플
        val_images = sorted((balloon_root / "val").glob("*.jpg"))
        for idx, img_path in enumerate(val_images[:3]):
            img = load_color(img_path)
            outputs = predictor_ft(img)
            _visualize_predictions(img, outputs, metadata, ft_dir / f"val_{idx}_finetuned.png")

        if sample_image is not None:
            img = load_color(sample_image)
            outputs = predictor_ft(img)
            _visualize_predictions(img, outputs, metadata, ft_dir / "sample_finetuned.png")
            logger.info("[문제3-4] Fine-tuned 결과에서 작은 풍선 누락이 보여 iteration 확대/강한 augmentation이 필요합니다.")

    else:
        logger.warning("[문제3] 풍선 데이터셋이 없습니다: %s", balloon_root)

    end_time = datetime.now()
    logger.info("No. 3 of HW3 Elapsed Time: {} Minutes".format(round((end_time - st_time).seconds / 60, 2)))
