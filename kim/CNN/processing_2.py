import os
import pandas as pd
from datetime import datetime


dataset_path = r"D:\dataset" # 원본 데이터셋 경로로 데이터 셋 저장되어 있는 위치로 설정

new_dataset_path = os.path.join(dataset_path, "created_train_data") # 새로운 데이터셋 경로로 저장할 위치랑 파일 이름 설정할 것

# 필요한 디렉토리 생성
if not os.path.exists(new_dataset_path):
    os.makedirs(new_dataset_path)

# 사용자 그룹 목록
user_groups = ["user01-06", "user07-10", "user11-12", "user21-25", "user26-30"]

# 각 사용자 그룹 내의 실제 사용자 폴더를 생성하고, 필요한 하위 디렉토리도 생성
for group in user_groups:
    group_path = os.path.join(dataset_path, group)
    for user in os.listdir(group_path):
        user_path = os.path.join(new_dataset_path, user)
        os.makedirs(user_path, exist_ok=True)

        subdirs = ["e4Acc", "e4Bvp", "e4Eda", "e4Hr", "e4Temp", "mAcc", "mGps", "mGyr", "mMag"]
        for subdir in subdirs:
            os.makedirs(os.path.join(user_path, subdir), exist_ok=True)

# 사용자별 데이터 병합
for group in user_groups:
    group_path = os.path.join(dataset_path, group)

    for user in os.listdir(group_path):
        user_orig_path = os.path.join(group_path, user)
        user_new_path = os.path.join(new_dataset_path, user)

        # Unix_epoch_timestamp 디렉토리 내의 데이터 탐색
        for day_dir in os.listdir(user_orig_path):
            day_path = os.path.join(user_orig_path, day_dir)

            if os.path.isdir(day_path):
                # Unix timestamp를 년-월-일 형식으로 변환
                try:
                    day_str = datetime.utcfromtimestamp(int(day_dir)).strftime('%Y-%m-%d')
                except ValueError:
                    continue

                # 날짜별로 데이터 프레임을 저장할 딕셔너리 생성
                dfs = {}

                for subdir in subdirs:
                    subdir_path = os.path.join(day_path, subdir)

                    if os.path.exists(subdir_path):
                        # 하위 디렉토리 내의 CSV 파일을 읽어 하나의 데이터프레임으로 병합
                        csv_files = [os.path.join(subdir_path, file) for file in os.listdir(subdir_path) if
                                     file.endswith('.csv')]
                        if csv_files:
                            combined_df = pd.concat([pd.read_csv(csv_file) for csv_file in csv_files],
                                                    ignore_index=True)
                            dfs[subdir] = combined_df

                # 날짜별로 데이터 프레임을 병합하여 CSV 파일로 저장
                for subdir, df in dfs.items():
                    combined_csv_path = os.path.join(user_new_path, subdir, f"{day_str}_{subdir}.csv")

                    # 기존에 해당 날짜의 CSV 파일이 있는지 확인하고, 있다면 읽어와서 기존 데이터프레임에 추가
                    if os.path.exists(combined_csv_path):
                        existing_df = pd.read_csv(combined_csv_path)
                        df = pd.concat([existing_df, df], ignore_index=True)

                    df.to_csv(combined_csv_path, index=False)

print("데이터 병합 완료")
