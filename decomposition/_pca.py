"""
@File   :_pca.py
@Author :JohsuaWu1997
@Date   :16/10/2020
"""
import torch
import numpy as np
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PCA:
    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit_tranform(self, X):
        U, S, V = self.fit(X, self.n_components)
        U = U[:, :self.n_components]
        return U

    def fit(self, X, n_components):
        n_samples, n_features = X.shape
        X -= torch.mean(X, dim=0)

        U, S, V = torch.svd(X)
        components_ = V

        # Get variance explained by singular values
        explained_variance_ = (S ** 2) / (n_samples - 1)
        total_var = explained_variance_.sum()
        explained_variance_ratio_ = explained_variance_ / total_var

        self.n_samples_, self.n_features_ = n_samples, n_features
        self.components_ = components_[:n_components]
        self.n_components_ = n_components
        self.explained_variance_ = explained_variance_[:n_components]
        self.explained_variance_ratio_ = explained_variance_ratio_[:n_components]
        return U, S, V
