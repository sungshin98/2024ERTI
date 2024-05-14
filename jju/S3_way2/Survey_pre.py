import pandas as pd
from os import path
from sklearn.preprocessing import MinMaxScaler

# 파일 로드
data_path = "D:\\ETRI"
survey_path = path.join(data_path, "user_survey_2020.csv")
data = pd.read_csv(survey_path)

# 필요한 열 선택
columns = ['userId', 'date', 'caffeine', 'cAmount(ml)', 'alcohol', 'aAmount(ml)']
data_selected = data[columns]

# 결측치 처리
data_selected = data_selected.fillna(0)

# 원-핫 인코딩 적용
data_encoded2_2 = pd.get_dummies(data_selected, columns=['caffeine', 'alcohol'])

# 카페인 종류와 알코올 종류
caffeine_types = [col for col in data_encoded2_2.columns if 'caffeine_' in col]
alcohol_types = [col for col in data_encoded2_2.columns if 'alcohol_' in col]

# 각 종류별로 사용자별 Min-Max 스케일링
for ctype in caffeine_types:
    for user in data_encoded2_2['userId'].unique():
        user_mask = data_encoded2_2['userId'] == user
        user_data = data_encoded2_2.loc[user_mask, ctype]
        if user_data.sum() > 0:  # 해당 사용자가 음료를 소비한 경우에만 스케일링
            scaler = MinMaxScaler()
            data_encoded2_2.loc[user_mask, f'{ctype}_amount_scaled'] = scaler.fit_transform(
                data_encoded2_2.loc[user_mask, ['cAmount(ml)']])
        else:
            data_encoded2_2.loc[user_mask, f'{ctype}_amount_scaled'] = 0

for atype in alcohol_types:
    for user in data_encoded2_2['userId'].unique():
        user_mask = data_encoded2_2['userId'] == user
        user_data = data_encoded2_2.loc[user_mask, atype]
        if user_data.sum() > 0:  # 해당 사용자가 음료를 소비한 경우에만 스케일링
            scaler = MinMaxScaler()
            data_encoded2_2.loc[user_mask, f'{atype}_amount_scaled'] = scaler.fit_transform(
                data_encoded2_2.loc[user_mask, ['aAmount(ml)']])
        else:
            data_encoded2_2.loc[user_mask, f'{atype}_amount_scaled'] = 0

S3Survey_path = path.join(data_path, "processed_user_survey_2020.csv")
data_encoded2_2.to_csv(S3Survey_path, index=False)

# 결과 확인
print(data_encoded2_2.info())