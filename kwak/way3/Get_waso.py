import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import conf_file

info_data = pd.read_csv(conf_file.created_info_path)
cls_data = pd.DataFrame()
cls_data[['userId', 'cluster']] = info_data[['userId', 'cluster']]
sleep_data = pd.read_csv(conf_file.sleep_path)
sleep_data = sleep_data[['userId', 'wakeupcount', 'wakeupduration']]
cls_data = pd.merge(cls_data, sleep_data, on='userId')

survey_data = pd.read_csv(conf_file.survey_path)
survey_data['cAmount(ml)'] = survey_data['cAmount(ml)'].fillna(0)
survey_data['aAmount(ml)'] = survey_data['aAmount(ml)'].fillna(0)
scaler = MinMaxScaler()
survey_data['cAmount'] = survey_data.groupby('userId')['cAmount(ml)'].transform(lambda x: (x - x.min()) / (x.max() - x.min()) if not x.empty else 0)
survey_data['aAmount'] = survey_data.groupby('userId')['aAmount(ml)'].transform(lambda x: (x - x.min()) / (x.max() - x.min()) if not x.empty else 0)
survey_data['cAmount'] = survey_data['cAmount'].fillna(0)
survey_data['aAmount'] = survey_data['aAmount'].fillna(0)
survey_data.drop(['cAmount(ml)', 'cAmount(ml)'], axis=1, inplace=True)

caffeine_encoded = pd.get_dummies(survey_data['caffeine'], prefix='caffeine')
alcohol_encoded = pd.get_dummies(survey_data['alcohol'], prefix='alcohol')
survey_data = pd.concat([survey_data, caffeine_encoded, alcohol_encoded], axis=1)
survey_data.drop(['caffeine', 'alcohol'], axis=1, inplace=True)
