import pandas as pd
from os import path
from sklearn.preprocessing import MinMaxScaler

# 파일 로드
data_path = "D:\\ETRI"
survey_path = path.join(data_path, "user_survey_2020.csv")
data = pd.read_csv(survey_path)

# 필요한 열 선택
columns = ['userId', 'caffeine', 'cAmount(ml)', 'alcohol', 'aAmount(ml)']
data_selected = data[columns]

# 결측치 처리
data_selected = data_selected.fillna(0)  # 변경된 부분

# 원-핫 인코딩 적용
data_encoded2 = pd.get_dummies(data_selected, columns=['caffeine', 'alcohol'])

# Min-Max 스케일링 초기화
scaler = MinMaxScaler()

# 카페인 종류와 알코올 종류
caffeine_types = [col for col in data_encoded2.columns if 'caffeine_' in col]
alcohol_types = [col for col in data_encoded2.columns if 'alcohol_' in col]

# 각 종류별 양을 스케일링
for ctype in caffeine_types:
    scaled_col_name = f'{ctype}_amount_scaled'
    data_encoded2[scaled_col_name] = scaler.fit_transform(
        data_encoded2[['cAmount(ml)']].where(data_encoded2[ctype] == 1))
    data_encoded2[scaled_col_name].fillna(0, inplace=True)

for atype in alcohol_types:
    scaled_col_name = f'{atype}_amount_scaled'
    data_encoded2[scaled_col_name] = scaler.fit_transform(
        data_encoded2[['aAmount(ml)']].where(data_encoded2[atype] == 1))
    data_encoded2[scaled_col_name].fillna(0, inplace=True)

# 결과 확인
print(data_encoded2.info())