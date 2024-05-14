import pandas as pd
import os
import os.path as path
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

data_path = "D:\\ETRI"
info_path = path.join(data_path, "user_info_2020.csv")

info_data = pd.read_csv(info_path)
info_data = info_data.drop(['handed', 'startDt', 'endDt'], axis=1)
data_to_scale = info_data[['age', 'height', 'weight']]

scaler = MinMaxScaler()
info_data[['age', 'height', 'weight']] = scaler.fit_transform(data_to_scale)
one_hot_encoded_gender = pd.get_dummies(info_data['gender'])
info_data = pd.concat([info_data, one_hot_encoded_gender], axis=1)
info_data.drop('gender', axis=1, inplace=True)

label = info_data['userId']
info_data.drop('userId', axis=1, inplace=True)

gmm = GaussianMixture(n_components=4, random_state=42)
clusters = gmm.fit_predict(info_data)

info_data['cluster'] = clusters
print("Clustered Info Data:\n", info_data)

pca = PCA(n_components=2)
info_data_2d = pca.fit_transform(info_data)

plt.figure(figsize=(8, 6))
for cluster_label in range(4):
    plt.scatter(info_data_2d[clusters == cluster_label, 0], info_data_2d[clusters == cluster_label, 1], label=f'Cluster {cluster_label}')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('GMM Clustering Result with Cluster Centers')
plt.legend()
plt.show()

info_data['userId'] = label
output_path = path.join(data_path, "created_data", "clustered_info_gmm.csv")
if os.path.exists(output_path):
    os.remove(output_path)
info_data.to_csv(output_path, index=False)