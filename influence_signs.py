import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.linear_model import LogisticRegression

# Построение тепловой карты средних значений признаков по кластерам
def plot_cluster_profiles(X, labels):
    df = X.copy()
    prof = df.assign(cluster=labels).groupby("cluster").mean().T

    plt.figure(figsize=(1.2*prof.shape[0], prof.shape[1]))
    ax = sns.heatmap(prof, cmap="magma_r")

    TITLE_FS = 10
    LABEL_FS = 10
    TICK_FS  = 8

    #ax.set_title("Тепловая карта средних значений признаков по кластерам", fontsize=TITLE_FS)
    ax.set_xlabel("Кластеры", fontsize=LABEL_FS)
    ax.set_ylabel("Признаки", fontsize=LABEL_FS)

    ax.tick_params(axis="x", labelsize=TICK_FS)
    ax.tick_params(axis="y", labelsize=TICK_FS)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=TICK_FS)

    plt.tight_layout()
    return prof


# Влияние признаков на кластеризацию на основе LogReg
def importance_logreg(X, labels, feature_names):
    clf = LogisticRegression(max_iter=2000, multi_class="auto", n_jobs=None)
    clf.fit(X, labels)
    imp = np.mean(np.abs(clf.coef_), axis=0)
    s = pd.Series(imp, index=feature_names).sort_values(ascending=True)

    plt.figure(figsize=(8, 0.35*len(feature_names)+0.5))
    s.plot(kind="barh")
    #plt.title("Важность признаков при кластеризации")
    plt.xlabel("Важность")
    plt.tight_layout()
    return s