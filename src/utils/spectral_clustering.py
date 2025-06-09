import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_kernels


def spectral_clustering(X, n_clusters=2, metric='rbf', gamma=1.0):
    """
    X: matriz de dados (n_samples, n_features)
    n_clusters: número de clusters desejados
    metric: métrica de similaridade ('rbf', 'cosine', etc.)
    gamma: parâmetro para kernel RBF
    """
    # 1. Matriz de similaridade
    #W = pairwise_kernels(X, metric=metric, gamma=gamma)
    W = X
    # 2. Matriz Laplaciana normalizada
    D = np.diag(W.sum(axis=1))
    D_inv_sqrt = np.linalg.inv(np.sqrt(D))
    L = np.eye(len(X)) - D_inv_sqrt @ W @ D_inv_sqrt

    # 3. Autovalores e autovetores
    eigvals, eigvecs = np.linalg.eigh(L)
    indices = np.argsort(eigvals)[:n_clusters]
    U = eigvecs[:, indices]

    # 4. Normalizar linhas
    U_norm = U / np.linalg.norm(U, axis=1)[:, np.newaxis]

    # 5. Aplicar k-means
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(U_norm)

    return clusters