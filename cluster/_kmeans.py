"""
@File   :_kmeans.py
@Author :JohsuaWu1997
@Date   :16/10/2020
"""
import torch
import numpy as np
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _k_init(X, n_clusters, init):
    # 1. Randomly choose clusters
    centers = torch.zeros((n_clusters, X.shape[1]), device=device)
    dist = torch.zeros((n_clusters, X.shape[0]), device=device)
    centers[0] = X.index_select(0, init)
    dist[0] = (centers[0] - X).norm(dim=1)
    for i in range(1, n_clusters):
        select = torch.argmax(torch.min(dist[:i], dim=0).values)
        centers[i] = X.index_select(0, select)
        dist[i] = (centers[i] - X).norm(dim=1)
    return centers


def _update_centers(X, labels, centers):
    select = torch.zeros((centers.shape[0], X.shape[0]), device=device)
    select[labels, torch.arange(X.shape[0])] = 1
    select = select[select.sum(dim=1) > 0]
    return torch.mm(select, X) / select.sum(dim=1, keepdim=True)


def _label_inertia(X, labels, centers):
    return (X - torch.index_select(centers, 0, labels)).norm(dim=1).mean()


class KMeans:
    def __init__(self, n_clusters, random_state, n_init=10, max_iter=300, tol=1e-4):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.labels_ = None
        self.centers_ = None
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        random.seed(self.random_state)

    def fit(self, X):
        inits = torch.randint(X.shape[0], (self.n_init,), device=device)
        best_inertia = None
        best_cluster = 0

        for init in inits:
            # 1. Randomly choose clusters
            centers = _k_init(X, self.n_clusters, init)

            for _iter in range(self.max_iter):
                # 2a. Assign labels based on closest center
                labels = torch.argmin(torch.cdist(X, centers), dim=1)

                # 2b. Find new centers from means of points
                new_centers = _update_centers(X, labels, centers)

                # 2c. Check for convergence
                if centers.shape[0] == new_centers.shape[0]:
                    if ((new_centers - centers).norm(dim=1) < self.tol).sum() == centers.shape[0]:
                        break
                centers = new_centers

            inertia = _label_inertia(X, labels, centers)
            if (
                    best_inertia is None or best_cluster < centers.shape[0] or
                    (inertia < best_inertia and best_cluster == centers.shape[0])
            ):
                best_inertia = inertia.clone().detach()
                best_cluster = centers.shape[0]
                self.labels_ = labels.clone().detach()
                self.centers_ = centers.clone().detach()

    def fit_predict(self, X, cuda=False):
        self.fit(X)
        return self.labels_ if cuda else self.labels_.cpu().numpy()
