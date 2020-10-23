"""
@File   :_kmeans.py
@Author :JohsuaWu1997
@Date   :16/10/2020
"""
import torch
import numpy as np
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
        self.n_samples = None
        self.n_features = None
        self.max_iter = max_iter
        self.tol = tol
        self.labels_ = None
        self.centers_ = None
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        random.seed(self.random_state)

    def _k_init(self, X):
        select = torch.randint(X.shape[0], (self.n_init,), device=device)
        centers = torch.zeros((self.n_clusters, self.n_init, self.n_features), device=device)
        dist = torch.zeros((self.n_clusters, self.n_init, self.n_samples), device=device)

        if self.n_init * self.n_clusters > self.n_samples:
            X_dist = torch.cdist(X, X)

        for i in range(self.n_clusters):
            centers[i] = X.index_select(0, select)
            if self.n_init * self.n_clusters > self.n_samples:
                dist[i] = torch.index_select(X_dist, 0, select)
            else:
                dist[i] = torch.cdist(centers[i], X)
            if i == 0:
                minimum = dist[0]
            else:
                minimum = torch.min(dist[i], minimum)
            select = torch.argmax(minimum, dim=1)
        return centers.transpose(0, 1)

    def fit(self, X):
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]

        centers = self._k_init(X)

        x = torch.arange(self.n_init, device=device).repeat_interleave(self.n_samples)
        y = torch.arange(self.n_samples, device=device).repeat(self.n_init)
        for _iter in range(self.max_iter):
            labels = torch.argmin(torch.cdist(centers, X), dim=1)
            select = torch.zeros(self.n_init, self.n_clusters, self.n_samples, device=device)
            select[x, labels.view(-1), y] = 1
            new_centers = torch.matmul(select, X) / select.sum(dim=2, keepdim=True)
            # new_centers = torch.where(new_centers.isnan(), centers, new_centers)
            select = select.sum(dim=2, keepdim=True).expand(self.n_init, self.n_clusters, self.n_features)
            new_centers = torch.where(select > 0, new_centers, centers)

            if ((new_centers - centers).norm(dim=2) < self.tol).sum() == centers.shape[0] * centers.shape[1]:
                break
            centers = new_centers.detach().clone()

        inertia = torch.stack([
            (X - torch.index_select(centers[i], 0, labels[i])).norm(dim=1).mean() for i in range(self.n_init)
        ])
        select = torch.argmin(inertia)
        self.labels_ = labels[select]

    def fit_predict(self, X, cuda=False):
        self.fit(X)
        return self.labels_ if cuda else self.labels_.cpu().numpy()
