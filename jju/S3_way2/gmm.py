import pandas as pd
import os
import os.path as path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pandas import DataFrame

data_path = "D:\\ETRI"
info_path = path.join(data_path, "user_info_2020.csv")
sleep_path = path.join(data_path, "user_sleep_2020.csv")

info_data = pd.read_csv(info_path)
sleep_data = pd.read_csv(sleep_path)
info_data = pd.merge(info_data, sleep_data[['userId', 'durationtosleep']], on='userId', how='left')

info_data = info_data.drop(['handed', 'startDt', 'endDt'], axis=1)
data_to_scale = info_data[['age', 'height', 'weight', 'durationtosleep']]
encoder = LabelEncoder()
info_data['gender'] = encoder.fit_transform(info_data['gender'])
numerical_data = info_data.select_dtypes(include=['float64', 'int64'])

scaler = StandardScaler()
scaled_data = scaler.fit_transform(numerical_data)

pca = PCA(n_components=4)
pca_result = pca.fit_transform(scaled_data)

f = plt.figure(figsize=(10, 8), dpi=80)
for i in range(2, 6):
    gm = GaussianMixture(n_components=i, random_state=42)
    pred = gm.fit_predict(scaled_data)
    centers = gm.means_
    pca_centers = pca.transform(centers)

    df = DataFrame({'x': pca_result[:, 0], 'y': pca_result[:, 1], 'label': pred})
    groups = df.groupby('label')

    ax = f.add_subplot(2, 2, i - 1)
    for name, group in groups:
        ax.scatter(group.x, group.y, label=name, s=8)
    ax.scatter(pca_centers[:, 0], pca_centers[:, 1], c='red', s=30, marker='X')
    ax.set_title(f"Cluster size: {i}")
    ax.legend()

plt.tight_layout()
plt.show()

output_path = path.join(data_path, "created_data", "clustered_info_gmm.csv")
try:
    if os.path.exists(output_path):
        os.remove(output_path)
    info_data.to_csv(output_path, index=False)
except Exception as e:
    print(f"Error occurred while trying to write to {output_path}: {e}")