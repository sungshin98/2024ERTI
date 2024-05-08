import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from kwak import conf_file as conf

pd.set_option('display.max_columns', None)

info_path = conf.info_path
user_info = pd.read_csv(info_path)
# 성별과 사용 손을 원-핫 인코딩으로 변환
#user_info_encoded = pd.get_dummies(user_info, columns=['gender', 'handed'], drop_first=False)
user_info_encoded = pd.get_dummies(user_info, columns=['gender'], drop_first=False)
# Min-Max 정규화를 위한 스케일러 생성
scaler = MinMaxScaler()
columns_to_normalize = ['age', 'height', 'weight']
user_info_encoded[columns_to_normalize] = scaler.fit_transform(user_info_encoded[columns_to_normalize])
user_info_processing = user_info_encoded.drop(columns=['startDt', 'endDt', 'handed'])
user_info = user_info_processing
