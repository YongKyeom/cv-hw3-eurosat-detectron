# HW3: Computer Vision Pipeline

성균관대학교 데이터사이언스융합학과 컴퓨터비전 수업의 과제 3번(HW3) 제출용 코드베이스입니다.
세 개의 주요 문제(BoF 기반 클래식 ML, EuroSAT 딥러닝 분류, Detectron2 Instance Segmentation)를
서로 다른 실행 스크립트에서 수행하도록 구성했습니다.

- 문제 1/2: `src/homework_03_0102.py`
- 문제 3: `src/homework_03_03.py`

각 문제는 동일한 루트 경로 설정 및 로거/시드를 공유합니다.

---

## 데이터 입력
- Scene-15: `data/SCENE-15/train`, `data/SCENE-15/test`
- EuroSAT: `data/EuroSAT/2750/...` (필요 시 자동 다운로드)
- Balloon: `data/balloon/train`, `data/balloon/val`

## 결과 출력
- 문제 1: `result/hw3/p1_*`
- 문제 2: `result/hw3/p2_*`
- 문제 3: `result/hw3/p3_detectron2/*`

로그 파일은 `log/` 디렉토리에 각각 `hw03_0102_*.log`, `hw03_03_*.log` 형태로 기록됩니다.

학과: 데이터사이언스융합학과
이름: 김용겸

---

## 문제별 상세

### 문제 1 — Bag-of-Features + Classical ML
- Scene-15 데이터셋을 격자형 키포인트로 분할하여 SIFT descriptor 추출
- Visual Codebook(K-Means) + BoFEncoder로 BoW 히스토그램 생성
- SVM / RandomForest / XGBoost baseline 학습 → Hyperopt로 파라미터 탐색, Confusion Matrix/HPO 로그 저장

### 문제 2 — EuroSAT 딥러닝 분류 (MLP / CNN / ResNet / ConvNeXt)
- torchvision.transforms 기반 전처리 및 train/val/test split
- 각 모델을 baseline → Hyperopt 순으로 학습하고, train/test Confusion Matrix와 Hyperopt 시각화 저장

### 문제 3 — Detectron2 Instance Segmentation Fine-tuning
- Mask R-CNN (R50-FPN)으로 balloon 데이터셋을 fine-tuning
- 사전학습/튜닝 모델 inference, COCO AP/AR 평가, 시각화 이미지 저장

---

## 실행 방법

### 문제 1, 2
```bash
cd /path/to/project
python src/homework_03_0102.py
```

### 문제 3
```bash
python src/homework_03_03.py
```

---

## 폴더 구조
```
project/
├── README.md
├── requirements.txt
├── log/
│   ├── hw03_0102_*.log         # 문제1/2 진행 로그
│   └── hw03_03_*.log           # 문제3 진행 로그
├── result/
│   ├── p1_svm/...
│   ├── p1_rf/...
│   ├── p1_xgb/...
│   ├── p2_mlp/...
│   ├── p2_cnn/...
│   ├── p2_resnet/...
│   ├── p2_convnext/...
│   └── p3_detectron2/...
├── data/
│   ├── SCENE-15/
│   │   ├── train/<class>/*.jpg
│   │   └── test/<class>/*.jpg
│   ├── EuroSAT/2750/<class>/*.jpg
│   └── balloon/
│       ├── train/*.jpg, via_region_data.json
│       └── val/*.jpg, via_region_data.json
└── src/
    ├── homework_03_0102.py
    ├── homework_03_03.py
    ├── bof/
    │   ├── codebook.py
    │   └── encoder.py
    ├── features/
    │   ├── descriptors.py
    │   └── patch.py
    ├── model/
    │   ├── dataset_eurosat.py
    │   ├── dl/
    │   │   ├── mlp.py 
    │   │   ├── cnn.py
    │   │   ├── resnet.py
    │   │   ├── convnext.py
    │   │   └── utils.py
    │   ├── ml/
    │   │   └── classical_ml.py
    │   ├── optim/
    │   │   └── hyperopt_runner.py
    │   └── trainer/
    │       ├── trainer_base.py
    │       ├── trainer_ml.py
    │       └── trainer_torch.py
    ├── detection/
    │   ├── balloon_dataset.py
    │   ├── config_builder.py
    │   ├── evaluator.py
    │   └── trainer_detectron.py
    └── utils/
        ├── io.py
        ├── logger.py
        ├── metric.py
        ├── paths.py
        └── visualize.py
```

---

## 의존성
```bash
pip install -r requirements.txt
```
