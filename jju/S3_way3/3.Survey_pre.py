import pandas as pd
from os import path
from sklearn.preprocessing import MinMaxScaler

data_path = "D:\\ETRI"
survey_path = path.join(data_path, "user_survey_2020.csv")
data = pd.read_csv(survey_path)

columns = ['userId', 'caffeine', 'cAmount(ml)', 'alcohol', 'aAmount(ml)']
data_selected = data[columns]

data_selected = data_selected.fillna(0)

data_encoded3 = pd.get_dummies(data_selected, columns=['caffeine', 'alcohol'])

scaler = MinMaxScaler()
caffeine_types = [col for col in data_encoded3.columns if 'caffeine_' in col]
alcohol_types = [col for col in data_encoded3.columns if 'alcohol_' in col]

for ctype in caffeine_types:
    data_encoded3[f'{ctype}_scaled'] = scaler.fit_transform(
        data_encoded3[['cAmount(ml)']].where(data_encoded3[ctype] == 1))
    data_encoded3[f'{ctype}_scaled'].fillna(0, inplace=True)

for atype in alcohol_types:
    data_encoded3[f'{atype}_scaled'] = scaler.fit_transform(
        data_encoded3[['aAmount(ml)']].where(data_encoded3[atype] == 1))
    data_encoded3[f'{atype}_scaled'].fillna(0, inplace=True)

grouped_data = {}
for ctype in caffeine_types:
    grouped_data[f'{ctype}_average_cAmount'] = data_encoded3[data_encoded3[ctype] == 1] \
        .groupby('userId')['cAmount(ml)'].mean()

for atype in alcohol_types:
    grouped_data[f'{atype}_average_aAmount'] = data_encoded3[data_encoded3[atype] == 1] \
        .groupby('userId')['aAmount(ml)'].mean()

grouped_df = pd.DataFrame(grouped_data)
print(grouped_df.head())