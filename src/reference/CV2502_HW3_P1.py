import time
import cv2
import glob
import os
import numpy as np
import random
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.kernel_approximation import AdditiveChi2Sampler
from sklearn.svm import LinearSVC

start_time = time.time()

'''
Parameters
'''
patch_stride = 16
K = 20

'''
Load Dataset
(don't modify)
'''
def scene15():
    train_folders = glob.glob("SCENE-15/train/*")
    train_folders.sort()
    classes = dict()
    x_train = list()
    y_train = list()
    for index, folder in enumerate(train_folders):
        label = os.path.basename(folder)
        classes[label] = index
        paths = glob.glob(os.path.join(folder, "*"))
        for path in paths:
            x_train.append(cv2.imread(path, 0))
            y_train.append(index)

    x_test = list()
    y_test = list()
    test_folders = glob.glob("SCENE-15/test/*")
    test_folders.sort()
    for folder in test_folders:
        label = os.path.basename(folder)
        index = classes[label]
        paths = glob.glob(os.path.join(folder, "*"))
        for path in paths:
            x_test.append(cv2.imread(path, 0))
            y_test.append(index)
    return x_train, y_train, x_test, y_test, sorted(classes.keys())

print("Load Dataset ...")
x_train, y_train, x_test, y_test, labels_names = scene15()
combined = list(zip(x_train, y_train))
random.shuffle(combined)
x_train[:], y_train[:] = zip(*combined)


'''
Extract Patches
(don't modify)
'''
train_key_points = list()
train_feature_shapes = list()
for image in x_train:
    h, w = image.shape
    image_key_points = list()
    for x in range(0, w, patch_stride):
        for y in range(0, h, patch_stride):
            image_key_points.append(cv2.KeyPoint(x, y, patch_stride))
    train_key_points.append(image_key_points)
    train_feature_shapes.append((len(range(0, w, patch_stride)), (len(range(0, h, patch_stride)))))

test_key_points = list()
test_feature_shapes = list()


for image in x_test:
    h, w = image.shape
    image_key_points = list()
    for x in range(0, w, patch_stride):
        for y in range(0, h, patch_stride):
            image_key_points.append(cv2.KeyPoint(x, y, patch_stride))
    test_key_points.append(image_key_points)
    test_feature_shapes.append((len(range(0, w, patch_stride)), (len(range(0, h, patch_stride)))))


'''
P-1.1
'''
###################################################################################
# 아래의 코드의 빈 곳(None 부분)을 채우세요.
# None 부분 외의 부분은 가급적 수정 하지 말고, 주어진 형식에 맞추어
# None 부분 만을 채워주세요. 임의적으로 전체적인 구조를 수정하셔도 좋지만,
# 파이썬 코딩에 익숙 하지 않으시면, 가급적 틀을 유지하시는 것을 권장합니다.
# 1) descriptor를 선정하세요. (SIFT, SURF 등) OpenCV의 패키지를 사용하시면 됩니다.
# 2) for 반복문 안에서, 1)에서 정의한 descriptor를 통하여 features를 추출하세요.
#    features의 차원은 (# of keypoints, feature_dim) 입니다.
###################################################################################




######## Write Your Code Here ##########
descriptor = None #Define your descriptor (e.g. surf, sift)

train_features = list()
index = 0
for image, key_points in zip(x_train, train_key_points):

    ######## Write Your Code Here ##########
    _, features = None
    ########################################


    train_features.append(features)
    index += 1

test_features = list()
index = 0
for image, key_points in zip(x_test, test_key_points):
    
    # Write Your Code Here #################
    _, features = None
    ########################################
    test_features.append(features)
    index += 1
    print("Extract Test Features ... {:4d}/{:4d}".format(index, len(x_test)))


'''
Normalizing
(don't modify)
'''
flattened_train_features = np.concatenate(train_features, axis=0)
pca = PCA(n_components=flattened_train_features.shape[-1], whiten=True)
pca.fit(flattened_train_features)
train_normalized_features = list()
index = 0
for features in train_features:
    features = pca.transform(features)
    train_normalized_features.append(features)
    index += 1
    print("Normalize Train Features ... {:4d}/{:4d}".format(index, len(train_features)))
test_normalized_features = list()
index = 0
for features in test_features:
    features = pca.transform(features)
    test_normalized_features.append(features)
    index += 1
    print("Normalize Test Features ... {:4d}/{:4d}".format(index, len(test_features)))



'''
P-1.2 :Make Codebook
'''

###################################################################################
# 아래의 코드의 빈 곳(None 부분)을 채우세요.
# None 부분 외의 부분은 가급적 수정 하지 말고, 주어진 형식에 맞추어 None 부분 만을 채워주세요
# 1) 함수 encode 부분 안의 None 부분을 채우세요.
#    distances는 K means 알고리즘을 통해 얻어진 centroids, 즉 codewords(visual words)와 각 이미지의 특징들 간의 거리 입니다.
#    distances 값을 이용하여, features(# of keypoints, feature_dim)를 인코딩(histogram 혹은 quantization이라고도 함) 하세요.    
#    인코딩된 결과인 representations은 K(centorid의 개수)로 표현되어야 합니다.
###################################################################################

class Codebook:

    def __init__(self, K):

        self.K = K

        self.kmeans = KMeans(n_clusters=K, verbose=True)

    def make_code_words(self, features):

        self.kmeans.fit(features)

    def encode(self, features, shapes):

        distances = self.kmeans.transform(features)
        representations = np.zeros(dtype=np.int64, shape=(len(distances), self.K))

        # Write Your Code Here ###################################################
        '''
        hint
        distance를 통해 representation을 이용할 떄, kmeans의 nearest centroid를 구하는 
        내장함수 or numpy.argmin, np.sum연산을 통해 구하실 수 있습니다.
        '''
    
        representations = None
        ##########################################################################

        if np.array(representations).shape != (self.K, ):
            # representations는 반드시 (K) 차원을 가져야 합니다.
            # 해당 조건문은 잘못 구현했을 경우를 판단하기 위해 작성되었으며, 추후 문제없이 구현되었다면 지우셔도 됩니다.
            print(np.array(representations).shape)
            print("Your code may be wrong")

        return representations

'''
Encode Codebook and encoded features 
(Don't modify)
'''
### CODE BOOK Make ####
print("Make Codebook ...")
flattened_normalized_train_features = pca.transform(flattened_train_features)
codebook = Codebook(K)
codebook.make_code_words(flattened_normalized_train_features)


train_encoded_features = list()
index = 0
for features, shapes in zip(train_normalized_features, train_feature_shapes):
    encoded_features = codebook.encode(features, shapes)
    train_encoded_features.append(encoded_features)
    index += 1
    print("Encoding Train Features ... {:4d}/{:4d}".format(index, len(train_normalized_features)))
test_encoded_features = list()
index = 0
for features, shapes in zip(test_normalized_features, test_feature_shapes):
    encoded_features = codebook.encode(features, shapes)
    test_encoded_features.append(encoded_features)
    index += 1
    print("Encoding Text Features ... {:4d}/{:4d}".format(index, len(test_normalized_features)))

'''
Approximate Kernel for encoded features
(Don't modify)
'''
chi2sampler = AdditiveChi2Sampler(sample_steps=2)
chi2sampler.fit(train_encoded_features, y_train)
train_encoded_features = chi2sampler.transform(train_encoded_features)
test_encoded_features = chi2sampler.transform(test_encoded_features)

'''


P-1.3 : Classify Images with SVM
'''
###################################################################################
# 아래의 코드의 빈 곳을 채우세요.
# 1) 아래의 model 부분에 sklearn 패키지를 활용하여, Linear SVM(SVC) 모델을 정의하세요.
#    처음에는 SVM의 parameter를 기본으로 설정하여 구동하시길 권장합니다.
#    구동 성공 시, SVM의 C 값과 max_iter 파라미터 등을 조정하여 성능 향상을 해보시길 바랍니다.
###################################################################################


#parameter 값을 수정해가면서 성능향상을 해보시길 바랍니다.
model = LinearSVC(C=None, max_iter=None, verbose=True)


print("Classify Images ...")
# Write Your Code Here ############################################################
'''
위에서 정의한 model을 None에 적절한 인수를 넣어 train data를 가지고 학습시켜보세요.
문제 1-3의 조건을 잘 읽고 해당되는 X와 y를 정의하면 됩니다(그냥 x_train 사용하는 것이 아님!).
'''

model.fit(None, y_train)

train_score = model.score(None, y_train)
test_score = model.score(None, y_test)

###################################################################################
elapsed_time = time.time() - start_time


'''
Print Results
'''
print()
print("=" * 90)
print("Train  Score: {:.5f}".format(train_score))
print("Test   Score: {:.5f}".format(test_score))
print("Elapsed Time: {:.2f} secs".format(elapsed_time))
print("=" * 90)

'''For confusion matrix'''
from sklearn.metrics import confusion_matrix
# Write Your Code Here ############################################################
'''
None에 적절한 코드를 넣어 Confusion matrix를 확인하고 visualize 해본 뒤 이에 대한 분석을 진행해주세요
'''

y_pred = model.predict(None)
Confusion_Matrix = None


###################################################################################

