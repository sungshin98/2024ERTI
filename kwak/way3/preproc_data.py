import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import conf_file

info_data = pd.read_csv(conf_file.created_info_path)
cls_data = pd.DataFrame()
cls_data[['userId', 'cluster']] = info_data[['userId', 'cluster']]
sleep_data = pd.read_csv(conf_file.sleep_path)
sleep_data = sleep_data[['userId', 'date', 'wakeupcount', 'wakeupduration']]
cls_data = pd.merge(sleep_data, cls_data, on='userId', how='inner')

survey_data = pd.read_csv(conf_file.survey_path)
survey_data = survey_data[survey_data['amPm'] != 'am']
# 필요없는 칼럼들 제거
survey_data.drop(columns=['amPm', 'startInput', 'endInput', 'sleep', 'sleepProblem', 'dream',
                          'amCondition', 'amEmotion', 'pmEmotion', 'pmStress', 'pmFatigue'], inplace=True)

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

merged_data = pd.merge(cls_data, survey_data, on=['userId', 'date'], how='inner')

label_data = pd.read_csv(conf_file.train_path)
label_data = label_data[['subject_id', 'date', 'S4']]
label_data['userId'] = label_data['subject_id']
label_data.drop(['subject_id'], axis=1, inplace=True)

merged_data = pd.merge(merged_data, label_data, on=['userId', 'date'], how='inner')
merged_data.to_csv('./preproc.csv', index=False)
