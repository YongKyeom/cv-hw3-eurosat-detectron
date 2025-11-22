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

## 환경 구성

### 문제 3-1/3-2 (.venv)
```bash
python3 -m venv .venv
source .venv/bin/activate  # Mac/Linux
# Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 문제 3-3 (.venv2, Detectron2 빌드)
- Mac 로컬에서 Detectron2 빌드를 위해 별도 가상환경 `.venv2` 사용
- PyTorch 1.9 계열 + numpy 1.x + OpenCV 4.9 + clang 빌드

```bash
python3 -m venv .venv2
source .venv2/bin/activate  # Mac/Linux
# Windows: .venv2\Scripts\activate

python -m pip install --upgrade pip setuptools wheel ninja
python -m pip install "torch==1.9.*" "torchvision==0.10.*" "torchaudio==0.9.*"
python -m pip install "numpy==1.26.4"
python -m pip install "opencv-python==4.9.0.80"
CC=clang CXX=clang++ ARCHFLAGS="-arch arm64" \
python -m pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'

pip install -r requirements2.txt
```

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
📂 project/
├── 📄 README.md
├── 📄 requirements.txt
├── 📂 log/
│   ├── 📄 hw03_0102_*.log              # 문제1/2 진행 로그
│   └── 📄 hw03_03_*.log                # 문제3 진행 로그
├── 📂 result/                          # 문항별 Output
│   ├── 📂 p1_svm/
│   ├── 📂 p1_rf/
│   ├── 📂 p1_xgb/
│   ├── 📂 p2_mlp/
│   ├── 📂 p2_cnn/
│   ├── 📂 p2_resnet/
│   ├── 📂 p2_convnext/
│   └── 📂 p3_detectron2/
├── 📂 data/                            # 데이터셋 모음
│   ├── 📂 SCENE-15/                    # Bag-of-Features용 Scene-15 (train/test)
│   ├── 📂 EuroSAT/2750/                # EuroSAT 10-class RGB 이미지
│   └── 📂 balloon/                     # Detectron2 Fine-tuning 풍선 데이터
└── 📂 src/                             # 실행 스크립트와 핵심 모듈
    ├── 📄 homework_03_0102.py          # **문제1/2 실행 스크립트**
    ├── 📄 homework_03_03.py            # **문제3 실행 스크립트**
    ├── 📂 bof/                         # BoF Codebook/Encoder
    │   ├── 📄 codebook.py
    │   └── 📄 encoder.py
    ├── 📂 features/                    # SIFT/HOG/패치 추출
    │   ├── 📄 descriptors.py
    │   └── 📄 patch.py
    ├── 📂 model/
    │   ├── 📄 dataset_eurosat.py        # 문제 2 데이터셋 Loader
    │   ├── 📂 dl/                       # 딥러닝 모델 아키텍처
    │   │   ├── 📄 mlp.py
    │   │   ├── 📄 cnn.py
    │   │   ├── 📄 resnet.py
    │   │   ├── 📄 convnext.py
    │   │   └── 📄 utils.py
    │   ├── 📂 ml/                       # SVM/RF/XGB 모델 아키텍처
    │   │   └── 📄 classical_ml.py
    │   ├── 📂 optim/                    # Hyper-parameter 최적화 모듈
    │   │   └── 📄 hyperopt_runner.py
    │   └── 📂 trainer/                  # ML/DL 모델 학습 실행 모듈
    │       ├── 📄 trainer_base.py
    │       ├── 📄 trainer_ml.py
    │       └── 📄 trainer_torch.py
    ├── 📂 detection/                    # Detectron2 Fine-tuning 구성
    │   ├── 📄 balloon_dataset.py
    │   ├── 📄 config_builder.py
    │   ├── 📄 evaluator.py
    │   ├── 📄 fine_tune_settings.py
    │   └── 📄 trainer_detectron.py
    └── 📂 utils/                        # 입출력/로그/지표/경로/시각화 공용 유틸
        ├── 📄 io.py
        ├── 📄 logger.py
        ├── 📄 metric.py
        ├── 📄 paths.py
        └── 📄 visualize.py
```

---
