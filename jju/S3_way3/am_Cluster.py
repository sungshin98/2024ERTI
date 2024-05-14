import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Define the path and load data
data_path = "D:\\ETRI"
survey_path = os.path.join(data_path, "user_survey_2020.csv")
data = pd.read_csv(survey_path)

# Filter for 'am' period data
am_data = data[data['amPm'] == 'am'].copy()

# Prepare data for clustering
cluster_data = am_data[['amCondition', 'amEmotion']]

# Scaling the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(cluster_data)

# GMM clustering
gmm = GaussianMixture(n_components=6, random_state=42)
clusters = gmm.fit_predict(scaled_data)

# Add cluster labels safely using .loc
am_data.loc[:, 'cluster'] = clusters  # This ensures that you're modifying the DataFrame directly

# Dimensionality reduction for visualization
pca = PCA(n_components=2)
data_2d = pca.fit_transform(scaled_data)

# Visualize the clustering results
plt.figure(figsize=(8, 6))
for cluster_label in range(6):
    cluster_points = data_2d[clusters == cluster_label]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster_label}')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('GMM Clustering Results with 6 Clusters')
plt.legend()
plt.show()

# Save the updated data
output_path = os.path.join(data_path, "clustered_am_data.csv")
am_data.to_csv(output_path, index=False)