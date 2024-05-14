import pandas as pd
from sklearn.preprocessing import StandardScaler
import conf_file

info_data = pd.read_csv(conf_file.created_info_path)
cls_data = pd.DataFrame()
cls_data[['userId', 'cluster']] = info_data[['userId', 'cluster']]
sleep_data = pd.read_csv(conf_file.sleep_path)
sleep_data = sleep_data[['userId', 'date', 'durationtosleep']]
cls_data = pd.merge(sleep_data, cls_data, on='userId', how='inner')

survey_data = pd.read_csv(conf_file.survey_path)
survey_data = survey_data[survey_data['amPm'] == 'pm']
survey_data.drop(columns=['amPm', 'startInput', 'endInput', 'sleep', 'sleepProblem', 'dream',
                          'amCondition', 'amEmotion'], inplace=True)  # 'pmCondition' 삭제 (선택적)

scaler = StandardScaler()
survey_data[['cAmount(ml)', 'aAmount(ml)']] = survey_data[['cAmount(ml)', 'aAmount(ml)']].fillna(0)
survey_data['cAmount'] = scaler.fit_transform(survey_data[['cAmount(ml)']])
survey_data['aAmount'] = scaler.fit_transform(survey_data[['aAmount(ml)']])
survey_data.drop(['cAmount(ml)', 'aAmount(ml)'], axis=1, inplace=True)

caffeine_encoded = pd.get_dummies(survey_data['caffeine'], prefix='caffeine')
alcohol_encoded = pd.get_dummies(survey_data['alcohol'], prefix='alcohol')
survey_data = pd.concat([survey_data, caffeine_encoded, alcohol_encoded], axis=1)
survey_data.drop(['caffeine', 'alcohol'], axis=1, inplace=True)

survey_data[['pmEmotion', 'pmStress', 'pmFatigue']] = scaler.fit_transform(survey_data[['pmEmotion', 'pmStress', 'pmFatigue']])

merged_data = pd.merge(cls_data, survey_data, on=['userId', 'date'], how='inner')

label_data = pd.read_csv(conf_file.train_path)
label_data = label_data[['subject_id', 'date', 'S3']]
label_data.rename(columns={'subject_id': 'userId'}, inplace=True)
merged_data = pd.merge(merged_data, label_data, on=['userId', 'date'], how='inner')

merged_data.to_csv('./merged.csv', index=False)
print(merged_data.head())