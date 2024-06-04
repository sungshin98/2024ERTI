import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 경로 설정
new_dataset_path = r"D:\dataset\created_train_data"
image_save_path = r"D:\dataset\images"

# 이미지 저장 경로가 없으면 생성
if not os.path.exists(image_save_path):
    os.makedirs(image_save_path)

# 사용자 목록과 서브디렉토리 목록
users = os.listdir(new_dataset_path)
subdirs = {
    "e4Acc": ["x", "y", "z"],
    "e4Bvp": ['timestamp',"value"],
    "e4Hr": ['timestamp',"hr"],
    "mAcc": ["x", "y", "z"],
    "mGps": ["lat", "lon"]
}

# Step 1: 데이터 시각화 및 이미지 저장
for user in users:
    user_path = os.path.join(new_dataset_path, user)
    for subdir, columns in subdirs.items():
        subdir_path = os.path.join(user_path, subdir)
        if os.path.exists(subdir_path):
            for file_name in os.listdir(subdir_path):
                if file_name.endswith('.csv'):
                    file_path = os.path.join(subdir_path, file_name)
                    data = pd.read_csv(file_path)

                    # 지정된 컬럼만 사용하여 데이터 추출
                    try:
                        data = data[columns]
                    except KeyError as e:
                        print(f"Error: Missing columns in {file_path}: {e}")
                        continue

                   # PCA를 적용할 수 있는지 확인
                    if data.shape[0] < 2 or data.shape[1] < 2:
                        print(f"Warning: Not enough samples or features in {file_path} for PCA.")
                        continue
                        # 차원 축소 (PCA 사용)
                    try:
                        pca = PCA(n_components=2)
                        reduced_data = pca.fit_transform(data)
                    except ValueError as e:
                        print(f"Error: PCA failed for {file_path}: {e}")
                        continue

                        # 데이터 플롯
                    plt.figure(figsize=(10, 4))
                    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], s = 1)
                    plt.title(f'{user} - {file_name}')
                    plt.xlabel('PCA Component 1')
                    plt.ylabel('PCA Component 2')
                    plt.grid(True)

                    # 이미지 저장
                    image_file_name = f"{user}_{file_name.replace('.csv', '.png')}"
                    image_file_path = os.path.join(image_save_path, image_file_name)
                    plt.savefig(image_file_path)
                    plt.close()

print("데이터 시각화 및 이미지 저장 완료")
