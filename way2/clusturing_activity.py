import conf
import pandas as pd
from os import path
import os
import datetime

data_path = conf.data_path
def find_user_folder(user_number):
    folders = ["user01-06", "user07-10", "user11-12", "user21-25", "user26-30"]
    user_number = int(user_number.split('user')[1])
    for folder in folders:
        start, end = folder.split("-")
        start = int(start[-2:])
        end = int(end[-2:])

        if start <= user_number <= end:
            user_folder = os.path.join(data_path, folder, f"user{user_number:02d}")
            if os.path.exists(user_folder):
                return user_folder
            else:
                print(f"User {user_number}의 데이터 폴더가 존재하지 않습니다.")
                return None

    print(f"User {user_number}는 주어진 범위 내에 존재하지 않습니다.")
    return None


def find_user_folder_by_real_time(real_time, user_folder):
    real_time = datetime.datetime.strptime(real_time, '%Y-%m-%d')
    target_date = real_time.strftime('%Y-%m-%d')
    folders = os.listdir(user_folder)
    for folder in folders:
        folder_epoch_time = int(folder)
        folder_date = datetime.datetime.utcfromtimestamp(folder_epoch_time).strftime('%Y-%m-%d')

        if folder_date == target_date:
            return folder
        else:
            print(f"No data folder found for the given date {target_date}.")
            return None

    print(f"No user folder found for the given real time {real_time}.")
    return None


def process_data(df):
    activities = [
        111, 112, 121, 122, 131, 132, 133, 134, 211, 212, 213, 22, 311, 312, 313, 314,
        321, 322, 41, 42, 43, 44, 45, 46, 51, 52, 53, 54, 55, 56, 57, 81, 82, 83, 84,
        85, 86, 87, 61, 62, 63, 64, 711, 712, 713, 721, 722, 723, 724, 725, 741, 742,
        743, 744, 745, 746, 751, 752, 753, 754, 755, 756, 761, 762, 763, 764, 791, 792,
        793, 91, 92
    ]
    action_df = pd.DataFrame(columns=activities)
    userIds = df['userId'].tolist()  # userId를 리스트로 변환
    dates = df['date'].tolist()  # date를 리스트로 변환
    for user_number, date in zip(userIds, dates):
        user_folder = find_user_folder(user_number)
        last_folder = find_user_folder_by_real_time(date, user_folder)


# get info_data
info_path = path.join(data_path,'user_info_2020.csv')
info_data = pd.read_csv(info_path)
info_data = info_data.drop(['handed', 'startDt', 'endDt'], axis=1)
info_data = pd.get_dummies(info_data, columns=['gender'])


# get sleep_data
sleep_path = path.join(data_path, 'user_sleep_2020.csv')
sleep_data = pd.read_csv(sleep_path)
selected_columns = ['timezone', 'startDt', 'endDt', 'lastUpdate', 'wakeupduration', 'lightsleepduration', 'deepsleepduration', 'wakeupcount', 'durationtosleep', 'remsleepduration', 'durationtowakeup', 'hr_average', 'hr_min', 'hr_max', 'rr_average', 'rr_min', 'rr_max', 'breathing_disturbances_intensity', 'snoring', 'snoringepisodecount']
sleep_data = sleep_data.drop(columns=selected_columns)


# get merged_data
merged_data = pd.merge(sleep_data, info_data, on='userId', how='left')
process_data(merged_data)