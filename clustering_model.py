import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from faiss_indexing import faiss_index
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from sklearn.decomposition import PCA
from sklearn.manifold import SpectralEmbedding

AX_LABEL_FS = 16
PCA_TITLE_FS = 22
PCA_POINT_SIZE = 85

# Рисует точки на плоскости PCA(2), раскрашенные по кластерам, и подписывает центроиды (средние по точкам кластера в 2D)
def plot_clusters_pca2(ax, X, labels, n_clusters, palette,
                       #title="Иерархическая кластеризация на PCA(2)",
                       annotate_centroids=True,
                       point_size=PCA_POINT_SIZE,
                       label_fs=AX_LABEL_FS,
                       title_fs=PCA_TITLE_FS):
    ax.scatter(
        X[:, 0], X[:, 1],
        s=point_size, c=palette[labels], alpha=0.9,
        marker="o", linewidths=0.3, edgecolor="k"
    )

    # центроиды
    centroids = []
    for i in range(n_clusters):
        mask = (labels == i)
        centroids.append(X[mask].mean(axis=0) if np.any(mask) else np.array([np.nan, np.nan]))
    centroids = np.vstack(centroids)

    valid = ~np.isnan(centroids).any(axis=1)
    if np.any(valid):
        ax.scatter(centroids[valid, 0], centroids[valid, 1],
                   marker="X", c="black", s=260, edgecolor="k", linewidths=0.5)
        if annotate_centroids:
            for i, (x, y) in enumerate(centroids):
                if np.isfinite(x) and np.isfinite(y):
                    ax.text(x, y, f"{i}", ha="center", va="center",
                            color="white", fontsize=14, weight="bold")

    #ax.set_title(title, fontsize=title_fs)
    ax.set_xlabel("PCA1", fontsize=label_fs)
    ax.set_ylabel("PCA2", fontsize=label_fs)
    ax.tick_params(axis="both", labelsize=label_fs)


def clusterize(X, n_clusters):
    connectivity = faiss_index(X)

    # Иерархическая кластеризация
    clusterer_finall = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage="ward",
        metric="euclidean",
        connectivity=connectivity,
        compute_distances=False
    )

    cluster_labels = clusterer_finall.fit_predict(X)

    X2 = PCA(n_components=2, random_state=0).fit_transform(X)
    #X3 = SpectralEmbedding(n_components=2, n_neighbors=20, random_state=0).fit_transform(X)

    fig, ax = plt.subplots(figsize=(26, 16))

    palette = np.array(sns.color_palette("magma", n_colors=n_clusters))

    # 2D-визуализации
    plot_clusters_pca2(ax=ax, X=X2, labels=cluster_labels, n_clusters=n_clusters, palette=palette
                       #title="2D визуализация иерархической кластеризации"
                       )

    plt.show()


# Построение дендрограммы
def plot_dendrogram(model, **kwargs):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    dendrogram(linkage_matrix, **kwargs)

def full_clusterize(X, p):
    connectivity = faiss_index(X)

    full_model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=0,
        linkage="ward",
        metric="euclidean",
        connectivity=connectivity,
        compute_distances=True
    ).fit(X)

    plt.figure(figsize=(min(60, max(10, X.shape[0] * 0.07)), 30))
    plot_dendrogram(full_model, truncate_mode="level", p=p)

    ax = plt.gca()
    #ax.set_title("Дендрограмма", fontsize=65)
    ax.set_xlabel("Листья (объекты)", fontsize=48)
    ax.set_ylabel("Дистанция", fontsize=48)
    ax.tick_params(axis='x', labelbottom=False)

    for line in ax.lines:
        line.set_linewidth(5.0)
    for coll in ax.collections:
        if hasattr(coll, "set_linewidths"):
            coll.set_linewidths([5.0])

    plt.tight_layout()
    plt.show()

