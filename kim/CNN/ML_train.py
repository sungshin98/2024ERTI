import os
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# 경로 설정 -> 이미지 저장했던 주소와 훈련 레이블의 주소, 모델 저장할 주소로 설정
image_save_path = r"D:\dataset\images"
label_file_path = r"D:\dataset\train_label.csv"
model_save_path = r"D:\dataset\model"

# 모델 저장 경로가 없으면 생성
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

# 레이블 파일 읽기
labels_df = pd.read_csv(label_file_path)

# 서브디렉토리 목록
subdirs = ['e4Acc', 'e4Bvp', 'e4Hr', 'mAcc', 'mGps']

# 이미지 파일 경로와 레이블 매칭
data = []
for user_name in labels_df['subject_id'].unique():
    for date in labels_df[labels_df['subject_id'] == user_name]['date'].unique():
        images = []
        all_subdirs_present = True
        for subdir in subdirs:
            image_file = f"{user_name}_{date}_{subdir}.png"
            image_path = os.path.join(image_save_path, image_file)
            if os.path.exists(image_path):
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 흑백 이미지로 읽기
                image = cv2.resize(image, (64, 64))
                images.append(image)
            else:
                all_subdirs_present = False
                break
        if all_subdirs_present:
            images = np.stack(images, axis=-1)  # 5개의 이미지를 채널 축으로 쌓기
            label_row = labels_df[(labels_df['subject_id'] == user_name) & (labels_df['date'] == date)]
            labels = label_row.iloc[0][['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']].values.astype(np.float32)
            data.append((images, labels))

# 데이터셋 준비 완료
print("이미지 파일 경로와 레이블 매칭 완료")

# 데이터셋 분리
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# 데이터 제너레이터 설정
datagen = ImageDataGenerator(rescale=1. / 255)


def data_generator(data, batch_size=32):
    while True:
        for start in range(0, len(data), batch_size):
            end = min(start + batch_size, len(data))
            batch_data = data[start:end]

            images = []
            labels = []

            for image, label in batch_data:
                images.append(image)
                labels.append(label)

            yield np.array(images, dtype=np.float32), np.array(labels, dtype=np.float32)


train_gen = data_generator(train_data)
test_gen = data_generator(test_data)


# VGG16 모델 정의
def build_vgg16_model(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),

        # Block 1
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2), strides=(2, 2)),

        # Block 2
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2), strides=(2, 2)),

        # Block 3
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2), strides=(2, 2)),

        # Block 4
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2), strides=(2, 2)),

        # Block 5
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2), strides=(2, 2)),

        # FC layers
        Flatten(),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='sigmoid')  # num_classes should be 7 for your case
    ])

    return model


# 이미지 크기 및 채널 설정
height, width, channels = 64, 64, 5  # 5개의 채널로 변경
num_classes = 7

# VGG16 모델 생성
model = build_vgg16_model((height, width, channels), num_classes)

# 모델 컴파일
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# 모델 체크포인트 설정
checkpoint = ModelCheckpoint(os.path.join(model_save_path, 'best_model_vgg16.keras'), monitor='val_accuracy',
                             save_best_only=True, mode='max')

# 모델 학습
model.fit(train_gen,
          steps_per_epoch=len(train_data) // 32,
          epochs=10,
          validation_data=test_gen,
          validation_steps=len(test_data) // 32,
          callbacks=[checkpoint])

# 최종 모델 저장
model.save(os.path.join(model_save_path, 'final_model_vgg16.keras'))

# 모델 평가
loss, accuracy = model.evaluate(test_gen, steps=len(test_data) // 32)
print(f"테스트 정확도: {accuracy}")
