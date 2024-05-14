import pandas as pd
import conf_file

info_data = pd.read_csv(conf_file.created_info_path)
sleep_data = pd.read_csv(conf_file.sleep_path)
survey_data = pd.read_csv(conf_file.S3Survey_path)
amCluster_data = pd.read_csv(conf_file.amCluster_path)
train_data = pd.read_csv(conf_file.train_path)

cls_data = info_data[['userId', 'cluster']]
cls2_data = amCluster_data[['userId', 'cluster']]
sleep_data = sleep_data[['userId', 'date', 'durationtosleep', 'lightsleepduration']]

merged_sleep_cls = pd.merge(sleep_data, cls_data, on='userId', how='inner')
merged_sleep_cls2 = pd.merge(merged_sleep_cls, cls2_data, on='userId', how='inner')

merged_survey_sleep_cls = pd.merge(merged_sleep_cls2, survey_data, on=['userId', 'date'], how='inner')

label_data = train_data[['subject_id', 'date', 'S3']]
label_data.rename(columns={'subject_id': 'userId'}, inplace=True)
final_merged_data3 = pd.merge(merged_survey_sleep_cls, label_data, on=['userId', 'date'], how='inner')

final_merged_data3.to_csv('./final_preproc3csv', index=False)
print(final_merged_data3.info())