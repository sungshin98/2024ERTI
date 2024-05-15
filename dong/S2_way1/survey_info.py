import pandas as pd

survey_df = pd.read_csv("D:/dataset/user_survey_2020.csv")


userIds = survey_df["userId"].unique() # survey의 userid 하나만 가져오기

for userId in userIds:
    idx_to_drop_am = survey_df[survey_df["userId"] == userId].index[0]  # 첫 번째 등장 행의 인덱스 가져오기
    survey_df.drop(idx_to_drop_am, inplace=True)  # 해당 행 삭제
    idx_to_drop_pm = survey_df[survey_df["userId"] == userId].index[-1]  # 마지막 등장 행의 인덱스 가져오기
    survey_df.drop(idx_to_drop_pm, inplace=True)  # 해당 행 삭제

survey_df.sort_values(by=["userId", "date", "amPm"], inplace=True)

survey_df.reset_index(drop=True, inplace=True)

survey_df.drop(columns=['dream', 'amEmotion', 'pmEmotion', 'pmStress', 'pmFatigue'], inplace=True)

survey_df[['sleep', 'sleepProblem', 'amCondition']] = survey_df[['sleep', 'sleepProblem', 'amCondition']].fillna(0)

# 홀수 행 선택
odd_rows = survey_df.iloc[::2]

# 짝수 행 선택
even_rows = survey_df.iloc[1::2]

# 2행씩 짝지어 더하기
new_columns = ['sleep', 'sleepProblem', 'amCondition']
new_values = even_rows[new_columns].reset_index(drop=True) + odd_rows[new_columns].reset_index(drop=True)

# 새로운 데이터프레임 생성
new_df = pd.DataFrame(new_values, columns=new_columns)

# userId, caffeine, cAmount(ml), alcohol, aAmount(ml) 열 선택하여 홀수 행에 추가
other_columns = ['userId', 'caffeine', 'cAmount(ml)', 'alcohol', 'aAmount(ml)']

new_df[other_columns] = odd_rows[other_columns].reset_index(drop=True)

# cAmount(ml)과 aAmount(ml)이 NaN인 데이터는 0으로 채우기
new_df["cAmount(ml)"] = new_df["cAmount(ml)"].fillna(0)
new_df["aAmount(ml)"] = new_df["aAmount(ml)"].fillna(0)

# caffeine, alcohol열은 각각 원-핫 인코딩을 해서 false면 0, true면 1로 하기
survey_reason = pd.get_dummies(new_df, columns=["caffeine", "alcohol"], dtype=int)
survey_reason = survey_reason[['userId'] + [col for col in survey_reason if col != 'userId']] #userId 맨 앞으로

# 결과 출력
print(new_df)
print("-------------------------------------------------------------------")
print(survey_reason)

survey_reason.to_csv(r"D:\dataset\created_data\survey_reason.csv", index=False)
