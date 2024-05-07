import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

import comb_data
import plot_silhouette

user_data = comb_data.user_data
plot_silhouette = plot_silhouette.plot_silhouette

# userId 열을 제외한 데이터프레임 준비
X = user_data.drop(columns=['userId', 'date', 'sleep'])
Repetition = 8
# K-means 모델 초기화 및 학습
for i in range(2, Repetition):
    # 주성분 분석을 통해 데이터를 2차원으로 축소
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    kmeans = KMeans(n_clusters=i, init='k-means++')  # 클러스터 개수는 적절하게 지정
    kmeans.fit(X_pca)

    # 클러스터링 결과 확인
    labels = kmeans.labels_
    print(labels)
    # 실루엣 계수 계산
    silhouette_avg = silhouette_score(X_pca, labels)
    print(f"{i}번째 전체 데이터의 평균 실루엣 계수:", silhouette_avg)

    cluster_centers = kmeans.cluster_centers_

    # 클러스터 중심점을 시각화
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.5)
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='x', s=100, c='red',
                label='Cluster Centers')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('K-means Clustering Result with Cluster Centers')
    plt.legend()
    plt.show()
    plot_silhouette(X_pca, labels)
    if i == Repetition-1:
        print(kmeans.labels_)
        X['cluster_label'] = kmeans.labels_
        X.to_csv('X_data.csv', index=False)