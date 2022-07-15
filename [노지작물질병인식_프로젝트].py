import math
import numpy as np
import tensorflow as tf
import cv2 as cv
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.optimizers import Adam
import os
import json
from PIL import Image
from tensorflow.keras.preprocessing import image

# 라벨 갯수
no_label = 21

# json 파일(label 데이터) 불러오기
label_path = "../datasets/라벨링데이터/"

label_list = os.listdir(label_path)
label_list_json = [json for json in label_list if json.endswith(".json")]
    
# 이미지 파일(input 데이터) 불러오기 
image_path = "../datasets/원천데이터/"

img_list = os.listdir(image_path)
img_list_jpg = [img for img in img_list if img.endswith(".jpg") or img.endswith(".JPG") or img.endswith(".jpeg") or img.endswith(".JPEG")]

# 훈련 집합 저장
x_train, y_train = [], []
count = 0
for i in range(300):
    # 이미지 추가
    img = Image.open(image_path+img_list_jpg[i]) # index에 해당하는 img 가져오기
    img = image.img_to_array(img) # Array로 변환
    img = cv.resize(img, dsize=(224, 224)) # ResNet 형태로 바꿔줌
    img = preprocess_input(img) # ResNet 모델에서 사용할 수 있도록 전처리
    x_train.append(img)
    
    # 라벨 추가
    label = label_list_json[i]
    with open(label_path+label) as file:
        data = json.load(file)
        label = data['annotations']['disease']
        y_train.append(label)

# 테스트 집합 저장
x_test, y_test = [], []
for i in range(0, 300, 10):
    # 이미지 추가
    img = Image.open(image_path+img_list_jpg[i]) # index에 해당하는 img 가져오기
    img = image.img_to_array(img) # Array로 변환
    img = cv.resize(img, dsize=(224, 224)) # ResNet 형태로 바꿔줌
    img = preprocess_input(img) # ResNet 모델에서 사용할 수 있도록 전처리
    x_test.append(img)
    
    # 라벨 추가
    label = label_list_json[i]
    with open(label_path+label) as file:
        data = json.load(file)
        label = data['annotations']['disease']
        y_test.append(label)
        
# tensorflow 에서 사용할 수 있도록 조작
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)
y_train = tf.keras.utils.to_categorical(y_train, no_label)
y_test = tf.keras.utils.to_categorical(y_test, no_label)

# ResNet에서 특징 추출 부분만 가져옴 (전이 학습)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(x_train[0].shape))

# 뒷 부분을 새로 부착
cnn = Sequential()
cnn.add(base_model)
cnn.add(Dropout(0.25))
cnn.add(Flatten())
cnn.add(Dense(1024, activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(no_label, activation='softmax')) 

# 미세 조정 방식의 학습(낮은 학습률 설정)
cnn.compile(loss='categorical_crossentropy', optimizer=Adam(0.00002), metrics=['accuracy'])
hist = cnn.fit(x_train, y_train, batch_size=16, epochs=2, validation_data=(x_test, y_test), verbose=1)

# 학습된 모델로 예측
res = cnn.evaluate(x_test, y_test, verbose=0)
print("정확률은 ", res[1]*100)

# 그래프 출력
import matplotlib.pyplot as plt

# 정확률 곡선
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='best')
plt.grid()
plt.show()

# 손실 함수 곡선
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='best')
plt.grid()
plt.show()




















