import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler

import conf_file as conf

pd.set_option('display.max_columns', None)


survey_path = conf.survet_path
user_survey = pd.read_csv(survey_path)
# 'amPm' 열의 값이 'am'인 행 선택
user_survey_am = user_survey[user_survey['amPm'] == 'am']
user_survey_am = user_survey_am.drop(columns=['amPm','startInput', 'endInput'])
user_survey_am = user_survey_am.drop(columns=['pmEmotion', 'pmStress', 'pmFatigue', 'caffeine', 'cAmount(ml)', 'alcohol', 'aAmount(ml)'])

# 'amPm' 열의 값이 'pm'인 행 선택
user_survey_pm = user_survey[user_survey['amPm'] == 'pm']
user_survey_pm = user_survey_pm.drop(columns=['amPm','startInput', 'endInput'])
user_survey_pm = user_survey_pm.drop(columns=['sleep', 'sleepProblem', 'dream', 'amCondition', 'amEmotion'])

#pm의 설문조사는 am 설문조사에 영향을 줄 것으로 예상됨
#date를 +1하여 합침
user_survey_pm['date'] = pd.to_datetime(user_survey_pm['date'])
user_survey_pm['date'] = user_survey_pm['date'] + timedelta(days=1)
user_survey_pm['date'] = user_survey_pm['date'].dt.strftime('%Y-%m-%d')

user_survey_comp = pd.merge(user_survey_am, user_survey_pm, on=['userId', 'date'], how='inner')

"""user_survey_comp['comb_caffeine'] = user_survey_comp['caffeine'] + ':' + user_survey_comp['cAmount(ml)'].astype(str)
user_survey_comp['comb_alcohol'] = user_survey_comp['alcohol'] + ':' + user_survey_comp['aAmount(ml)'].astype(str)
user_survey_comp = pd.get_dummies(user_survey_comp, columns=['comb_caffeine', 'comb_alcohol'], drop_first=False, prefix=['', ''])
user_survey_comp = user_survey_comp.drop(columns=['caffeine','alcohol', 'cAmount(ml)', 'aAmount(ml)'])
"""

user_survey_comp = pd.get_dummies(user_survey_comp, columns=['caffeine', 'alcohol'], drop_first=False, prefix=['', ''])
user_survey_comp = user_survey_comp.drop(columns=['cAmount(ml)', 'aAmount(ml)'])

user_survey = user_survey_comp


# Min-max scaler 초기화
scaler = MinMaxScaler()
columns_to_scale = ['sleep', 'sleepProblem', 'dream', 'amCondition', 'amEmotion', 'pmEmotion', 'pmStress', 'pmFatigue']
user_survey[columns_to_scale] = scaler.fit_transform(user_survey[columns_to_scale])
print(user_survey.shape)