"""
@File   :t_sne.py
@Author :JohsuaWu1997
@Date   :16/10/2020
"""
import torch
import numpy as np
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TSNE_NN(torch.nn.Module):
    def __init__(self):
        pass


class TSNE:
    def __init__(self, n_components=2, perplexity=30, lr=1e-3, n_iter=1000):
        self.n_components = n_components
        self.perplexity = perplexity
        self.lr = lr
        self.n_iter = n_iter

    def fit(self, X):
        pass

    def fit_transform(self, X):
        pass
