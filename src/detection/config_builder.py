"""
Detectron2 Config 생성 모듈: Mask R-CNN 모델을 선택해 config 생성
"""

from __future__ import annotations

from typing import Optional

from detectron2 import model_zoo
from detectron2.config import get_cfg


def build_config(
    model_name: str,
    train_dataset: str,
    output_dir: str = "./detectron2_output",
    base_lr: float = 0.00025,
    max_iter: int = 1000,
    ims_per_batch: int = 2,
    num_classes: Optional[int] = 1,
    device: str = "cpu",
) -> "CfgNode":
    """
    Mask R-CNN 등 Detectron2 모델을 Fine-tuning하기 위한 config 생성.

    Args:
        model_name (str): model_zoo config yaml 이름 (예: "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml")
        train_dataset (str): DatasetCatalog 이름 (예: "balloon_train")
        output_dir (str): 학습 결과 저장 경로
        base_lr (float): Learning rate
        max_iter (int): Max iteration
        ims_per_batch (int): batch size

        num_classes (Optional[int]): ROI Head 클래스 수. None이면 model_zoo 설정을 유지.
        device (str): 사용할 디바이스 이름(e.g., "cpu", "cuda").

    Returns:
        CfgNode: Detectron2 config 객체
    """
    cfg = get_cfg()
    cfg.MODEL.DEVICE = device  # CPU 디바이스
    cfg.merge_from_file(model_zoo.get_config_file(model_name))  # 모델 구조 로드

    cfg.DATASETS.TRAIN = (train_dataset,)  # 학습 데이터셋 지정
    cfg.DATASETS.TEST = ()  # 평가 데이터셋은 외부에서 세팅
    cfg.DATALOADER.NUM_WORKERS = 4  # DataLoader 워커 수

    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_name)  # 사전학습 가중치 경로
    cfg.SOLVER.IMS_PER_BATCH = ims_per_batch  # 배치 크기
    cfg.SOLVER.BASE_LR = base_lr  # 러닝레이트
    cfg.SOLVER.MAX_ITER = max_iter  # 학습 iteration 수
    cfg.SOLVER.STEPS = []  # LR decay 비활성화

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64  # ROI Head 샘플 수
    if num_classes is not None:
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes  # 클래스 수 설정 (None이면 기본 유지)

    cfg.OUTPUT_DIR = output_dir  # 결과 저장 경로

    return cfg
