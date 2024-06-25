import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.src.utils import to_categorical

# 데이터 경로 설정
data_dir = r"C:\Users\user\Desktop\AI_test\lfw"
categories = os.listdir(data_dir)
num_classes = len(categories)

# 이미지와 레이블을 담을 리스트 초기화
images = []
labels = []

# 이미지를 로드하고 전처리
for category in categories:
    class_index = categories.index(category)
    path = os.path.join(data_dir, category)
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (64, 64))
        images.append(img)
        labels.append(class_index)

# 이미지와 레이블을 NumPy 배열로 변환
images = np.array(images, dtype='float32') / 255.0
labels = np.array(labels)

# 레이블을 원-핫 인코딩
labels = to_categorical(labels, num_classes)

# 훈련 데이터와 검증 데이터로 나누기
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

from keras.api.models import Sequential
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

model.save('face_recognition_model.h5')

# 클래스 인덱스를 저장
class_indices = {i: categories[i] for i in range(num_classes)}
import pickle
with open('class_indices.pkl', 'wb') as f:
    pickle.dump(class_indices, f)

import keras.src.saving

# 모델 저장 (Keras 포맷 사용)
keras.saving.save_model(model, 'face_recognition_model.keras')
