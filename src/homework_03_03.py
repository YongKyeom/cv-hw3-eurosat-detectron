# type: ignore
"""HW3 문제 3 전용 Detectron2 파이프라인.

Balloon 데이터셋으로 Mask R-CNN을 파인튜닝하고, 문제에서 요구하는
사전학습/튜닝 시각화 및 COCOEvaluator 결과를 생성한다.
"""

from __future__ import annotations

import logging
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

warnings.filterwarnings("ignore")

import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer

from detection.balloon_dataset import load_balloon_dataset
from detection.config_builder import build_config
from detection.evaluator import evaluate_model
from detection.fine_tune_settings import SWEEP_SETTINGS, FineTuneSetting
from detection.trainer_detectron import BalloonTrainer
from model.dl.utils import seed_everything
from utils.io import ensure_dir, find_first_image, load_color, save_image, save_json
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
    """Detectron2 추론 결과를 저장한다.

    Args:
        image_bgr: BGR 이미지 배열.
        outputs: predictor 호출 결과.
        metadata: Detectron2 Metadata.
        save_path: 저장 경로.
        scale: Visualizer 스케일.
    """

    vis = Visualizer(
        image_bgr[:, :, ::-1],
        metadata=metadata,
        scale=scale,
        instance_mode=ColorMode.IMAGE,
    )
    vis_output = vis.draw_instance_predictions(outputs["instances"].to("cpu"))
    save_image(save_path, vis_output.get_image()[:, :, ::-1])


def _render_batch_predictions(
    predictor: DefaultPredictor,
    metadata: Any,
    image_paths: Sequence[Path],
    save_dir: Path,
    prefix: str,
) -> None:
    """여러 이미지를 순회하면서 추론 결과를 저장한다.

    Args:
        predictor: Detectron2 DefaultPredictor 객체.
        metadata: 시각화에 사용할 Metadata.
        image_paths: 추론할 이미지 경로 목록.
        save_dir: 이미지 저장 폴더.
        prefix: 저장 파일 접두어.
    """

    ensure_dir(save_dir)
    for idx, img_path in enumerate(image_paths):
        img = load_color(img_path)
        outputs = predictor(img)
        file_name = f"{prefix}_{idx:02d}_{img_path.stem}.png"
        _visualize_predictions(img, outputs, metadata, save_dir / file_name)


def _register_balloon_datasets(balloon_root: Path) -> None:
    """Balloon train/val을 Detectron2 Catalog에 등록한다."""

    for split in ["train", "val"]:
        name = f"balloon_{split}"
        if name not in DatasetCatalog.list():
            DatasetCatalog.register(
                name,
                lambda split=split: load_balloon_dataset(balloon_root / split),
            )
            MetadataCatalog.get(name).set(thing_classes=["balloon"])


def _select_sample_image(balloon_root: Path, assets_dir: Path) -> Optional[Path]:
    """샘플 이미지를 선택한다.

    우선순위: assets 폴더 → balloon/val → balloon/train.

    Args:
        balloon_root: balloon 데이터 루트 경로.
        assets_dir: 프로젝트 assets 폴더 경로.

    Returns:
        Optional[Path]: 선택된 이미지 경로. 없으면 None.
    """

    if assets_dir.exists():
        asset_sample = find_first_image(assets_dir)
        if asset_sample is not None:
            if logger:
                logger.info("[문제3] assets 샘플 이미지 사용: %s", asset_sample)
            return asset_sample

    fallback = find_first_image(balloon_root / "val") or find_first_image(balloon_root / "train")
    if fallback and logger:
        logger.info("[문제3] balloon 데이터셋 이미지 사용: %s", fallback)
    return fallback


def _collect_val_images(balloon_root: Path, limit: int) -> List[Path]:
    """val 폴더에서 시각화에 사용할 이미지를 모은다."""

    val_dir = balloon_root / "val"
    images = sorted(val_dir.glob("*.jpg"))
    return images[:limit]


def _log_dataset_stats(balloon_root: Path) -> None:
    """train/val 샘플 수를 로그로 남긴다."""

    for split in ["train", "val"]:
        size = len(load_balloon_dataset(balloon_root / split))
        if logger:
            logger.info("[문제3] %s samples: %d", split, size)


def _get_pretrained_metadata() -> Any:
    """COCO 80-class 시각화용 Metadata를 반환한다."""

    try:
        coco_meta = MetadataCatalog.get("coco_2017_val")
        if hasattr(coco_meta, "thing_classes") and len(coco_meta.thing_classes) >= 80:
            return coco_meta
    except KeyError:
        pass
    return MetadataCatalog.get("balloon_train")


if __name__ == "__main__":
    st_time = datetime.now()

    # ------------------------------------------------------------------
    # 0. 경로/로거/시드 설정
    # ------------------------------------------------------------------
    project_root = Path(os.path.join(os.path.dirname(__file__), "..")).resolve()
    paths = Paths.from_root(project_root)
    data_dir = paths.data_dir
    result_dir = paths.result_dir
    assets_dir = project_root / "assets"

    logger = get_logger(paths.log_dir, "hw03_03")
    logger.info("Project Root: %s", project_root)
    logger.info("Result Root: %s", result_dir)

    seed_everything(2025)

    # ==============================================================================
    # 문제 3: Detectron2 Segmentation + Fine-tunning
    # ==============================================================================
    # Detectron2 DatasetCatalog 등록 및 데이터 통계 로깅
    balloon_root = data_dir / "balloon"
    _register_balloon_datasets(balloon_root)
    _log_dataset_stats(balloon_root)

    metadata_balloon = MetadataCatalog.get("balloon_train")
    metadata_pretrained = _get_pretrained_metadata()

    # 사전/파인튜닝 결과 비교용 val 이미지 샘플
    val_images = _collect_val_images(balloon_root, 5)

    # Pre-trained 모델: 가벼운 Mask R-CNN (R50-FPN) 사용
    model_name = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"
    logger.info("[문제3] 모델 선택: %s → 가용한 자원 내 Fine-tunning 가능 모델", model_name)

    p3_dir = result_dir / "p3_detectron2"
    pre_dir = p3_dir / "pretrained"
    ft_dir = p3_dir / "finetuned"
    ensure_dir(pre_dir)
    ensure_dir(ft_dir)

    # 샘플 이미지 선택 (assets → val → train 순)
    sample_image = _select_sample_image(balloon_root, assets_dir)

    # ------------------------------------------------------------------
    # 3-1 사전학습 모델 확인
    #     • COCO 80-class 헤드 그대로 로드 → 예시/val 이미지 시각화
    # ------------------------------------------------------------------
    pre_vis_cfg = build_config(
        model_name=model_name,
        train_dataset="balloon_train",
        output_dir=str(pre_dir / "tmp_vis"),
        base_lr=0.00025,
        max_iter=1000,
        num_classes=None,
    )
    pre_vis_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    predictor_pre = DefaultPredictor(pre_vis_cfg)  # 사전학습 모델 인퍼런스 엔진

    if sample_image is not None:
        img = load_color(sample_image)
        outputs = predictor_pre(img)
        _visualize_predictions(img, outputs, metadata_pretrained, pre_dir / "sample_pretrained.png")

    if val_images:
        _render_batch_predictions(
            predictor_pre,
            metadata_pretrained,
            val_images,
            pre_dir / "val_pretrained",
            prefix="pre",
        )

    # ------------------------------------------------------------------
    # 3-3 모델 검증 (파인튜닝 전 baseline)
    #     • Balloon 1-class 헤드로 AP/AR baseline 계산 및 저장
    # ------------------------------------------------------------------
    pre_eval_cfg = build_config(
        model_name=model_name,
        train_dataset="balloon_train",
        output_dir=str(pre_dir / "baseline_eval"),
        base_lr=0.00025,
        max_iter=200,
        num_classes=1,
    )
    pre_eval_cfg.DATASETS.TEST = ("balloon_val",)  # 평가 데이터셋 지정
    pre_eval_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    pre_eval_metrics = evaluate_model(pre_eval_cfg, "balloon_val")  # 사전학습 모델 baseline 평가
    save_json({"pretrained": pre_eval_metrics}, pre_dir / "pretrained_metrics.json")
    logger.info("[문제3] Pre-trained Eval: %s", pre_eval_metrics)

    sweep_logs: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # 3-2/3-3/3-4 파인튜닝 및 비교
    #     • 스윕별 학습 → 평가(AP/AR) 저장 → val/samples 시각화
    # ------------------------------------------------------------------
    for setting in SWEEP_SETTINGS:
        # 파라미터 스윕별 결과 폴더 생성
        exp_dir = ft_dir / setting.label
        ensure_dir(exp_dir)

        ft_cfg = build_config(
            model_name=model_name,
            train_dataset="balloon_train",
            output_dir=str(exp_dir / "output"),
            base_lr=setting.base_lr,
            max_iter=setting.max_iter,
            ims_per_batch=setting.ims_per_batch,
            num_classes=1,
        )
        Path(ft_cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        trainer = BalloonTrainer(ft_cfg)  # Detectron2 DefaultTrainer 확장 (데이터 로더/학습 루프 포함)
        trainer.train_with_output()  # 지정한 파라미터로 파인튜닝 시작

        final_weights = Path(ft_cfg.OUTPUT_DIR) / "model_final.pth"
        ft_cfg.MODEL.WEIGHTS = str(final_weights)
        ft_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        ft_cfg.DATASETS.TEST = ("balloon_val",)
        predictor_ft = DefaultPredictor(ft_cfg)  # 파인튜닝된 모델 인퍼런스 엔진

        # 3-3) 파인튜닝 모델 성능 평가 (COCOEvaluator)
        ft_eval = evaluate_model(ft_cfg, "balloon_val", weights_path=str(final_weights))
        save_json(
            {
                "setting": setting.to_dict(),
                "metrics": ft_eval,
            },
            exp_dir / "finetuned_metrics.json",
        )
        logger.info("[문제3] Fine-tuned Eval (%s): %s", setting.label, ft_eval)
        sweep_logs.append({"setting": setting.to_dict(), "metrics": ft_eval})

        if val_images:
            _render_batch_predictions(
                predictor_ft,
                metadata_balloon,
                val_images,
                exp_dir / "val_finetuned",
                prefix="finetuned",
            )

        if sample_image is not None:
            img = load_color(sample_image)
            outputs = predictor_ft(img)
            _visualize_predictions(img, outputs, metadata_balloon, exp_dir / "sample_finetuned.png")
            logger.info(
                "[문제3-4] %s 샘플 재검증 완료 - iteration/augmentation 튜닝 필요 여부 확인",
                setting.label,
            )

    save_json(
        {
            "pretrained": pre_eval_metrics,
            "sweeps": sweep_logs,
        },
        p3_dir / "evaluation_summary.json",
    )

    end_time = datetime.now()
    logger.info(
        "No. 3 of HW3 Elapsed Time: %s Minutes",
        round((end_time - st_time).seconds / 60, 2),
    )
