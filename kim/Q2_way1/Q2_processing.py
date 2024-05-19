import Q2_file as fn
import pandas as pd

# 데이터프레임 읽기
df = pd.read_csv(fn.file_path)
clustering_df = pd.read_csv(fn.clustering_path)
reason_df = pd.read_csv(fn.survey_reason_path)

# 필요한 열 추출
selected_columns = ['userId', 'durationtosleep','lightsleepduration' ,'deepsleepduration', 'wakeupcount','snoring', 'snoringepisodecount', 'sleep_score']
selected_df = df[selected_columns]

# userId를 기준으로 병합
sleep_info = pd.merge(selected_df, clustering_df, on='userId')

# 두 데이터프레임을 병합합니다.
merged_df = pd.merge(reason_df, sleep_info, on='userId')

# userId 열을 user와 num으로 분리하고 num만 userId에 대체합니다.
merged_df[['user', 'num']] = merged_df['userId'].str.split('user', expand=True)
merged_df['userId'] = merged_df['num']

print(merged_df)
print(merged_df.dtypes)
merged_df.to_csv(r"D:\dataset\created_data\Q2_merged_df.csv", index=False)





