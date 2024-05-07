import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
import numpy as np


def plot_silhouette(X, cluster_labels):
    # 클러스터 수 구하기
    n_clusters = len(np.unique(cluster_labels))

    # 실루엣 계수 계산
    silhouette_avg = silhouette_score(X, cluster_labels)
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10

    # 각 클러스터에 대한 실루엣 계수 그리기
    for i in range(n_clusters):
        # 클러스터 i에 속하는 실루엣 계수 추출
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = plt.cm.nipy_spectral(float(i) / n_clusters)
        plt.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # 각 클러스터의 레이블 표시
        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # 다음 클러스터의 y_lower 조정
        y_lower = y_upper + 10

    plt.title("Silhouette plot for the various clusters")
    plt.xlabel("Silhouette coefficient values")
    plt.ylabel("Cluster label")

    # 전체 데이터의 실루엣 계수 표시
    plt.axvline(x=silhouette_avg, color="red", linestyle="--")
    plt.yticks([])
    plt.xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.show()

    return silhouette_avg