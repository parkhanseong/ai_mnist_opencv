# opencv 라이브러리 import
import cv2
from google.colab.patches import cv2_imshow
import numpy as np

# opencv로 이미지를 읽는다
img_input = cv2.imread("number_nine.png")

# 입력 이미지를 회색조로 변경하고, 28*28 이미지로 사이즈 변환
gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
gray = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
cv2_imshow(gray)

# 회색조로 바꾼 이미지를 임계값에 따라 흑백으로 변경
(thresh, img_binary) = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
cv2_imshow(img_binary)

# 현재 이미지의 높이와 너비에 따라 이미지 크기 조정
h,w = img_binary.shape

ratio = 100 / h
new_h = 100
new_w = w * ratio

img_temp = np.zeros((110,110), dtype=img_binary.dtype)
img_binary = cv2.resize(img_binary, (int(new_w), int(new_h)), interpolation=cv2.INTER_AREA)
img_temp[:img_binary.shape[0], :img_binary.shape[1]] = img_binary

img_binary = img_temp
cv2_imshow(img_binary)

# 이미지에서 폐곡선을 찾는다
cnts = cv2.findContours(img_binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 컨투어의 무게중심 좌표를 구합니다. 
M = cv2.moments(cnts[0][0])
center_x = (M["m10"] / M["m00"])
center_y = (M["m01"] / M["m00"])

# 무게 중심이 이미지 중심으로 오도록 이동시킵니다. 
height,width = img_binary.shape[:2]
shiftx = width/2-center_x
shifty = height/2-center_y

Translation_Matrix = np.float32([[1, 0, shiftx],[0, 1, shifty]])

# warpAffine 함수로 이미지를 이동
img_binary = cv2.warpAffine(img_binary, Translation_Matrix, (width,height))

img_binary = cv2.resize(img_binary, (28, 28), interpolation=cv2.INTER_AREA)
cv2_imshow(img_binary)

# model 에 넣기 전에 reshape
img_binary = img_binary.reshape(1, 28, 28, 1)

# Install TensorFlow
try:
  # %tensorflow_version only exists in Colab.
  %tensorflow_version 2.x
except Exception:
  pass

import tensorflow as tf
import keras

load_model = tf.keras.models.load_model("model.h5")

from google.colab import drive
drive.mount('/content/drive')

prediction = load_model.predict_classes(img_binary)
print(prediction[0])