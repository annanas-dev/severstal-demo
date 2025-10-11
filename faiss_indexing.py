import pandas as pd
import math
import numpy as np
import faiss
from scipy.sparse import csr_matrix


# Функция расчета параметров для IVFPQ
# Предлагает нам оптимальные параметры, опираясь на документацию и некоторые расчеты
# Но мы также можем самостоятельно задавать их значения
def pick_ivfpq_params(
        N, d, nprobe,
        m=None,  # кол-во субквантайзеров PQ
        C=None,  # коэффициент перед sqrt(N) для nlist
        k=None,
        nbits=8,
        nlist_min=64,  # кол-во инвертированных списков
        nlist_max=8192,
):
    if N <= 0 or d <= 0:
        raise ValueError("N и d должны быть > 0. В противном случае датафрейм пуст.")

    # 1) m
    if m is None:
        if d < 8:
            m = d
        else:
            m_choices = [8, 16, 32, 64]
            m = min(m_choices,
                    key=lambda x: abs((d / x) - 8))  # стремимся к тому, чтобы длина одного субвектора была ~8
    else:
        m = int(m)

    target_dimension = int(
        math.ceil(d / float(m)) * m)  # предпочтительная размерность вектора признаков, такая что % m == 0

    # 2) nlist
    C = 4.0 if C is None else float(C)
    if not (4.0 <= C <= 16.0):
        raise ValueError("C предпочтительней (согласно документации) в диапазоне [4, 16]")
    nlist_raw = int(max(1, C * math.sqrt(N)))
    nlist_raw = max(int(nlist_min), min(int(nlist_max), nlist_raw))
    nlist_raw = min(nlist_raw, int(N))
    # к ближайшей степени двойки
    a = int(round(math.log2(max(1, nlist_raw))))
    nlist = 2 ** a
    nlist = max(1, min(nlist, int(N)))

    # 3) проверка адекватности nprobe относительно k
    if k is None:
        k = int(np.sqrt(N))
    k = int(k)

    nprobe_assigned = nprobe
    avg_inv_list_len = max(1, int(round(N / nlist)))  # средняя длина инвертированного списка
    potentially_viewed_objects = int(avg_inv_list_len * nprobe)  # сколько объектов будет просмотрено (ориентировочно)
    recommended_nprobe_for_k = int(
        math.ceil(k / avg_inv_list_len))  # рекомендуемое значение nprobe, чтоб просмотерть k соседй
    is_rational = potentially_viewed_objects >= k  # хватает ли текущего nprobe, чтобы просмотреть k соседей
    if not is_rational:
        nprobe = recommended_nprobe_for_k

    return {
        "m": int(m),
        "target_dimension": int(target_dimension),
        "nbits": int(nbits),
        "C": float(C),
        "nlist": int(nlist),
        "nprobe_assigned": int(nprobe_assigned),
        "nprobe": int(nprobe),
        "k": k,
        "avg_list_len": int(avg_inv_list_len),
        "potentially_viewed_objects_with_nprobe": int(potentially_viewed_objects),
        "recommended_nprobe_for_k": int(recommended_nprobe_for_k),
        "nrobe_is_rational?": bool(is_rational)
    }


# Функция приведения размерность вектора к числу % m = 0
def pad_to_target_dimension(df, d, ivfpq_params, fill_value=0.0):
    m = ivfpq_params['m']  # кол-во квантайзеров
    target_dim = ivfpq_params['target_dimension']

    if target_dim > d:
        df = np.pad(df, ((0, 0), (0, ivfpq_params['target_dimension'] - d)), mode='constant', constant_values=0.0)
    d = df.shape[1]
    return df, d


# Функция для удаления элемента из списка его k ближайших соседей
def drop_self(D, I):
    N = I.shape[0]
    k = I.shape[1] - 1
    rows = np.arange(N)[:, None]
    mask = (I != rows)

    I_without_self = np.empty((N, k), dtype=I.dtype)
    D_without_self = np.empty((N, k), dtype=D.dtype)

    for i in range(N):
        idxs_without_self = np.flatnonzero(mask[i])
        I_without_self[i] = I[i, idxs_without_self]
        D_without_self[i] = D[i, idxs_without_self]

    return D_without_self, I_without_self


def faiss_index(df: pd.DataFrame):
    X = df.copy()

    n, d = X.shape

    ivfpq_params = pick_ivfpq_params(N=n, d=d, nprobe=16)

    X, d = pad_to_target_dimension(X, d, ivfpq_params)

    m = ivfpq_params['m']
    nlist = ivfpq_params['nlist']
    nbits = ivfpq_params['nbits']
    nprobe = ivfpq_params['nprobe']
    k = ['k']
    metric = faiss.METRIC_L2
    k = int(np.sqrt(n))

    quantiser = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFPQ(quantiser, d, nlist, m, nbits, metric)

    index.train(X)
    index.add(X)

    index.nprobe = nprobe

    D_with_self, I_with_self = index.search(X, k + 1)  # в каждом из инвертированных спсков будет содеражться и сам объект

    D_without_self, I_without_self = drop_self(D_with_self, I_with_self)

    # Ищем матрицу смежности, которая будет передана в алгоритм агломеративной кластеризации
    rows_all = np.repeat(np.arange(n), I_without_self.shape[1])
    cols_all = I_without_self.reshape(-1)
    valid = cols_all >= 0
    rows = rows_all[valid]
    cols = cols_all[valid]
    data = np.ones_like(cols, dtype=np.float32)
    adjacency_matrix = csr_matrix((data, (rows, cols)), shape=(n, n))
    symmetrical_adjacency_matrix = adjacency_matrix.maximum(adjacency_matrix.T).astype(bool)

    return symmetrical_adjacency_matrix