import pandas as pd
import os.path as path
from sklearn.preprocessing import MinMaxScaler
import os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

data_path = "G:\dataset"
info_path = path.join(data_path, "user_info_2020.csv")
survey_path = path.join(data_path,"user_survey_2020.csv")

info_data = pd.read_csv(info_path)
info_data = info_data.drop(['handed','startDt','endDt'], axis=1)
data_to_scale = info_data[['age', 'height', 'weight']]

scaler = MinMaxScaler()
info_data[['age', 'height', 'weight']] = scaler.fit_transform(data_to_scale)
one_hot_encoded_gender = pd.get_dummies(info_data['gender'])
info_data = pd.concat([info_data, one_hot_encoded_gender], axis=1)
info_data.drop('gender', axis=1, inplace=True)

label = info_data['userId']
info_data.drop('userId', axis=1, inplace=True)
# K-Means 모델 생성

kmeans = KMeans(n_clusters=4, random_state=42)

# 클러스터링 수행 및 결과 예측
clusters = kmeans.fit_predict(info_data)

# 클러스터링 결과를 DataFrame에 추가
info_data['cluster'] = clusters
print("Clustered Info Data:\n", info_data)

pca = PCA(n_components=2)

info_data_2d = pca.fit_transform(info_data)

# 클러스터링 결과를 시각화
plt.figure(figsize=(8, 6))

# 각 클러스터를 산점도로 플롯
for cluster_label in range(4):
    plt.scatter(info_data_2d[clusters == cluster_label, 0], info_data_2d[clusters == cluster_label, 1], label=f'Cluster {cluster_label}')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('K-means Clustering Result with Cluster Centers')
plt.legend()
plt.show()

info_data['userId'] = label
output_path = "G:\dataset\created_data\clusted_info.csv"
if os.path.exists(output_path):
    os.remove(output_path)
info_data.to_csv(output_path, index=False)