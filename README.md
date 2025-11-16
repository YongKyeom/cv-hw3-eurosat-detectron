# HW3: Computer Vision Pipeline

성균관대학교 데이터사이언스융합학과 컴퓨터비전 수업의 과제 3번(HW3) 제출용 코드베이스입니다. 
세 개의 주요 문제(BoF 기반 클래식 ML, EuroSAT 딥러닝 분류, Detectron2 Instance Segmentation)를 
하나의 실행 스크립트(`src/homework_03.py`)에서 순차적으로 수행하도록 구성했습니다.

- 코드: `src/homework_03.py`  
- 입력: 
  - `data/SCENE-15/train`, `data/SCENE-15/test`
  - `data/EuroSAT/2750/...` (필요 시 자동 다운로드)
  - `data/balloon/train`, `data/balloon/val`
- 출력: 
  - `result/hw3/p1_*`: BoF 히스토그램 PNG, Confusion Matrix, Hyperopt 로그/시각화, 모델 체크포인트
  - `result/hw3/p2_*`: Train/Test Confusion Matrix, Hyperopt 로그/시각화, DL 모델 체크포인트
  - `result/hw3/p3_detectron2`: Pre/Post 예측 시각화, COCO 평가 로그
- 로그: `log/hw03_*.log`: 모든 진행 상황(정확도/정밀도/F1/Confusion Matrix 범위 등)의 상세 로그  

학과: 데이터사이언스융합학과  
이름: 김용겸  

---

## 문제별 개요

### 문제 1. Bag-of-Features + Classical ML
- Scene-15 데이터셋을 사용하여 격자형 키포인트 + SIFT 특징을 추출합니다.
- Visual Codebook(K-Means) + BoF Encoder로 이미지마다 히스토그램 벡터를 생성합니다.
- SVM / Random Forest / XGBoost Baseline 학습 후 Hyperopt로 파라미터를 탐색합니다.
- 결과로 Confusion Matrix, BoF 히스토그램 샘플, Hyperopt 로그/히트맵/산점도 등을
  `result/hw3/p1_*` 위치에 저장합니다.

---

### 문제 2. EuroSAT 분류 (MLP / CNN / ResNet / ConvNeXt)
- torchvision.transforms로 32×32 리사이즈 및 augmentation 후 train/val/test 로더를 구성합니다.
- 각 모델(Multi-layer Perceptron, CNN, ResNet, ConvNeXt)을 Baseline → Hyperopt 순서로 학습합니다.
- Hyperopt는 layer 수, channel 성장률, dropout/lr 등 세부 파라미터를 탐색하며 결과를
  CSV/이미지로 저장합니다.
- Train/Test Confusion Matrix와 Accuracy/Precision/Recall/F1 로그를 기록하고,
  `result/hw3/p2_*`에 저장합니다.

---

### 문제 3. Detectron2 Instance Segmentation Fine-tuning
- Detectron2 Mask R-CNN (R50-FPN)을 balloon 데이터셋에 맞게 fine-tuning 합니다.
- 사전학습 모델 inference, fine-tuning 결과 inference, COCO AP/AR evaluation을 진행합니다.
- 시각화 이미지는 `result/hw3/p3_detectron2`에 저장됩니다.

---

## 사전 요구사항
- Python 3.12.7 권장
- 의존성 설치(`pip install -r requirements.txt`)
- SCENE-15, EuroSAT, balloon 데이터셋은 `data/` 하위에 위치해야 합니다.
  - `data/SCENE-15/train`, `data/SCENE-15/test`
  - `data/EuroSAT/2750/...` (필요 시 자동 다운로드)
  - `data/balloon/train`, `data/balloon/val`


## 실행 방법
```bash
cd src
python homework_03.py
```

모든 결과/로그는 `result/hw3/` 및 `log/` 디렉터리에 생성됩니다.


## 폴더 구조
```
project_folder
├── README.md
├── requirements.txt
├── log/
│   └── hw03_*.log
├── result/
│   └── hw3/
│       ├── p1_svm/
│       │   ├── confusion_baseline_svm.png
│       │   ├── confusion_tuned_svm.png
│       │   ├── hyperopt_logs_svm.csv
│       │   ├── hyperopt_plots/
│       │   └── svm_baseline.pkl / svm_tuned.pkl
│       ├── p1_rf/
│       ├── p1_xgb/
│       ├── p2_mlp/
│       │   ├── train_baseline_confusion.png
│       │   ├── test_baseline_confusion.png
│       │   ├── train_tuned_confusion.png
│       │   ├── test_tuned_confusion.png
│       │   ├── hyperopt_logs_mlp.csv
│       │   ├── hyperopt_plots/
│       │   └── mlp_baseline.pt / mlp_tuned.pt
│       ├── p2_cnn/
│       ├── p2_resnet/
│       ├── p2_convnext/
│       ├── p3_detectron2/
│       │   ├── pretrained/
│       │   │   └── sample_pretrained.png
│       │   └── finetuned/
│       │       ├── sample_finetuned.png
│       │       └── val_0_finetuned.png ...
│       ├── p1_metrics.csv
│       └── p2_metrics.csv
├── data/
│   ├── SCENE-15/
│   │   ├── train/<class>/*.jpg
│   │   └── test/<class>/*.jpg
│   ├── EuroSAT/2750/<class>/*.jpg
│   └── balloon/
│       ├── train/*.jpg / via_region_data.json
│       └── val/*.jpg / via_region_data.json
└── src/
    ├── homework_03.py
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
    └── utils/
        ├── io.py
        ├── logger.py
        ├── metric.py
        ├── paths.py
        └── visualize.py
```
