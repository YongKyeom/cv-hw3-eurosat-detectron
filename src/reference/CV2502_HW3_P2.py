import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from torchvision.datasets import ImageFolder, utils
from torchvision.transforms import transforms
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import seaborn as sns

#EuroSAT 데이터셋 다운로드 코드
#만약 다운로드 코드에 에러가 발생한다면, 아래 링크에서 직접 다운로드 후 압축을 해제해 주세요.
def get_EuroSAT(dirname):
    import os
    if(os.path.exists(dirname)):
        print("Dataset is already exist.")
        return(os.path.join(dirname, '2750'))
    
    os.makedirs(dirname, exist_ok=True)
    utils.download_and_extract_archive(
                "http://madm.dfki.de/files/sentinel/EuroSAT.zip",
                download_root=dirname,
                md5="c8fa014336c82ac7804f0398fcb19387",
                remove_finished=True,
            )
    return(os.path.join(dirname, '2750'))

# GPU 사용이 가능하다면 사용하고, 그렇지 않다면 CPU 사용
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

'''P-2.1'''

'''
data를 변형시키는 augmentation (rotation, color)을 추가하는 것은 가능합니다.
사용하신다면 리포트에 관련 내용및 사용한 이유를 적어주세요.
'''

# 데이터 전처리 및 로드
transform = transforms.Compose([
    #---------------------------------------------------#
    #여기에 이미지 크기를 32 x 32로 변형하는 코드를 작성해 주세요
    #---------------------------------------------------#
   
    transforms.ToTensor(),  # 텐서로 변환
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

dataset = ImageFolder(get_EuroSAT('EuroSAT'), transform=transform)


# 학습, 검증, 테스트 데이터 분할
dataset_size = len(dataset)
print('Dataset size:', dataset_size)
indices = list(range(dataset_size))

#아래 split에서 데이터셋을 적절하게 train, val, test로 분할하세요.
#코드를 보고 None에 적절한 인수값을 넣어 dataset을 분리시키세요

split1 = int(np.floor(None * dataset_size))
split2 = int(np.floor(None * dataset_size))

np.random.shuffle(indices)
train_indices, val_indices, test_indices = indices[:split1], indices[split1:split2], indices[split2:]

train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)
test_dataset = Subset(dataset, test_indices)

# 데이터 로더 설정
train_loader = DataLoader(None, batch_size=64, shuffle=True)
val_loader = DataLoader(None, batch_size=64, shuffle=False)
test_loader = DataLoader(None, batch_size=64, shuffle=False)

print('Train loader size:', len(train_loader.dataset))
print('Val loader size:', len(val_loader.dataset))
print('Test loader size:', len(test_loader.dataset))


'''P-2.2'''
# MLP 모델 정의
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        ### write your code####

        #######################
        self.activation = None

    def forward(self, x):

        x = x.view(x.size(0), -1)
        '''
        forward 함수는 torch.nn.module이 input x를 받아 output을 진행하는 함수입니다.
        참조 : (https://pytorch.org/docs/stable/generated/torch.nn.Module.html) 참조
        model의 비선형을 위해 꼭 활성화 함수를 사용해 주셔야 합니다. 
        활성화 함수는 https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity 를 참조해 보세요.
        '''
        ### write your code####

        #######################


        return x

'''P-2.3'''
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        ### Write your code ####
        

    def forward(self, x):
        ### write your code ####
        '''
        forward 함수는 torch.nn.module이 input x를 받아 output을 진행하는 함수입니다.
        참조 : (https://pytorch.org/docs/stable/generated/torch.nn.Module.html) 참조
        model의 비선형을 위해 꼭 활성화 함수를 사용해 주셔야 합니다. 
        활성화 함수는 https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity 를 참조해 보세요.
        '''
        #######################
        
        return x

# 모델 초기화 및 손실 함수, 최적화 알고리즘 설정
model = MLP().to(device) #CNN 사용시, CNN().to(device)



# loss objective : cross entropy 
# 다양한 손실 함수에 대해 알고싶으시다면 https://pytorch.org/docs/stable/nn.html#loss-functions 참고하세요.
criterion = nn.CrossEntropyLoss()

'''P-2.4'''

import torch.optim as optim
#여기에 최적화 함수를 정의하세요. https://pytorch.org/docs/stable/optim.html 참고하세요.
optimizer = None
print(model)

# 학습 함수
def train(model, train_loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
    train_loss = running_loss / len(train_loader.dataset)
    return train_loss


# 검증 함수
def validate(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            _, predictions = torch.max(outputs, 1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss = running_loss / len(val_loader.dataset)
    val_accuracy = accuracy_score(all_labels, all_predictions)
    return val_loss, val_accuracy


# 학습 및 검증 실행
#전체 epoch은 100으로 설정
num_epochs = 100

for epoch in range(num_epochs):
    train_loss = train(model, train_loader, optimizer, criterion)
    val_loss, val_accuracy = validate(model, val_loader, criterion)
    print(f"Iter {epoch + 1}/{num_epochs}, "
          f"Train Loss: {train_loss:.4f}, "
          f"Val Loss: {val_loss:.4f}, "
          f"Val Accuracy: {val_accuracy:.4f}")
 
# train, validate 함수를 수정하여 accuracy와 confusion matrix를 계산하세요.
# 예시인 sklearn의 confusion matrix함수는 다음과 같이 구현할 수 있습니다.

_, test_accuracy, test_confusion_matrix = validate(model, None, criterion)
print(f"Test Accuracy: {test_accuracy:.4f}")

# train 데이터에 대한 결과 출력
_, train_accuracy, train_confusion_matrix = validate(model, None, criterion)
print(f"Train Accuracy: {train_accuracy:.4f}")


# 이 confusion matrix의 시각화는 sklearn.metrics.ConfusionMatrixDisplay로 할 수도 있지만, seaborn을 사용한 예시는 다음과 같습니다.
# hmhm = sns.heatmap(ex_confusion_matrix, annot=True, fmt="d", cmap="Blues")
# hmhm.figure.savefig("output_ex_cm.png")
# hmhm.figure.clf()




