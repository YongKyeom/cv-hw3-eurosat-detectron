'''
#torch, cuda 버전을 확인하고 detectron2를 설치하세요. (https://detectron2.readthedocs.io/en/latest/tutorials/install.html)
#개인 환경에서 구현하시는 분들은 해당 명령어를 터미널에 작성하시면 됩니다.
#python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
#colab tutorial : https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5

'''

'''
예시 이미지와 풍선 데이터셋 다운로드
#개인 환경에서 구현하시는 분들은 !를 떼고 해당 명령어를 터미널에 작성하시면 됩니다.
# !wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O img_for_P3.jpg
# !wget https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip
# 이후 balloon dataset의 압축을 풀어주세요
'''

import torch
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)


# #colab에서 아래 코드를 실행시키는 경우 다음 함수로 시각화 하세요.
# from google.colab.patches import cv2_imshow

import cv2
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import warnings
warnings.filterwarnings(action='ignore')

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
import random
import os
import json
import numpy as np

cfg = get_cfg()

'''P3-1'''
# 과제 ppt를 참조하여 적절한 모델을 골라 아래 model_name을 해당하는 yaml 이름으로 변경하세요.
# ex) model_name = mask_rcnn_R_50_DC5_1x.yaml

model_name = None

cfg.merge_from_file(model_zoo.get_config_file(model_name))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_name)
predictor = DefaultPredictor(cfg)

# 예시 이미지를 불러와 확인해주세요 

img = cv2.imread('your download root', cv2.IMREAD_COLOR)
outputs = predictor(img)
v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

#확인 저장용 code
cv2.imwrite('your imwrite root',out.get_image()[:, :, ::-1])


'''Don't modify '''
# 풍선 데이터셋의 형식을 파인튜닝할 수 있게 만들어주는 함수입니다.
def get_balloon_dicts(img_dir):
    json_file = os.path.join(img_dir, "via_region_data.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        record = {}
        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
        annos = v["regions"]
        objs = []
        for _, anno in annos.items():
            assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]
            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

'''Don't modify '''
# 풍선 데이터셋을 DatasetCatalog에 등록하는 과정입니다.
for d in ["train", "val"]:
    if "balloon_" + d in DatasetCatalog.list():
        DatasetCatalog.remove("balloon_" + d)
    DatasetCatalog.register("balloon_" + d, lambda d=d: get_balloon_dicts("balloon/" + d))
    MetadataCatalog.get("balloon_" + d).set(thing_classes=["balloon"])
balloon_metadata = MetadataCatalog.get("balloon_train")
dataset_dicts = get_balloon_dicts("balloon/train")

# 풍선 데이터셋을 pre-trained 모델에 넣어 확인해보는 과정입니다.

'''
d["file_name"]으로부터 파일명만 추출하여 저장하고 싶을 경우 
import os
filename = os.path.basename(d["file_name"])
cv2.imwrite(f'./yourroot/{filename}',out.get_image()[:, :, ::-1])
'''

random.seed(1234)
for d in random.sample(dataset_dicts, 15):
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    v = Visualizer(im[:, :, ::-1],
                   MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
                   scale=0.5
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    ### only modify ####
    cv2.imwrite('your imwrite root',out.get_image()[:, :, ::-1])


'''P3-2,3'''
###Finetuning #####

# 파인튜닝을 위해 config를 수정하는 과정입니다.
# 여러 parameter를 변경하여 fine-tuning을 해볼 수 있습니다.
# None 부분을 수정하여 fine-tuning을 진행해주세요

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(model_name))
cfg.DATASETS.TRAIN = ("balloon_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_name)    # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = None                                 # This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.BASE_LR = None                                       # pick a good LR, if very low or high value => model cannot learn well
cfg.SOLVER.MAX_ITER = None                                      # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []                                           # decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128                  # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1                             # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)

# 파인튜닝 트레이너를 정의하는 과정입니다.
# None 부분을 document를 확인하여 채워주세요
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = None(cfg) #https://detectron2.readthedocs.io/en/latest/tutorials/training.html 문서를 확인하여 trainder를 정의하세요.
trainer.resume_or_load(resume=False)
trainer.train()

'''Don't modify '''
# 파인튜닝한 모델로 풍선 데이터셋을 확인해보는 과정입니다.
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
predictor = DefaultPredictor(cfg)

###########write your code#####################
# 여기에 모델 검증을 위한 코드를 작성하세요, 모델 검증은 balloon/val로 진행합니다.
# from detectron2.evaluation import COCOEvaluator, inference_on_dataset
# from detectron2.data import build_detection_test_loader
# 참고 문서들을 확인하여 구현하시면 됩니다.
# https://detectron2.readthedocs.io/en/latest/




###############################################



'''P3-4'''
# 앞서 불러온 예시 이미지를 finetuning이 완료된 모델에 적용해보세요
# 시각화 하였을 때, 위 방식 finetuning의 문제점 및 해결책에 대한 본인의 생각을 서술해보세요 
# HINT : finetuning 하는 코드를 잘 살펴보세요