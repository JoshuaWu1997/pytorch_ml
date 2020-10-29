"""
@File   :_vae.py
@Author :JohsuaWu1997
@Date   :17/10/2020
"""
import torch
import numpy as np
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _vae_loss_function(y_pred, x, mu, logvar):
    mse_loss = torch.nn.MSELoss(reduction='sum')
    reconstruction_loss = mse_loss(y_pred, x)
    KL_divergence = -0.5 * torch.sum(1 + logvar - torch.exp(logvar) - mu ** 2)
    return reconstruction_loss + KL_divergence


class VAE_NN(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(VAE_NN, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(D_in, D_in),
            torch.nn.ReLU(),
            torch.nn.Linear(D_in, H),
            torch.nn.ReLU(),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(H, D_out),
            torch.nn.ReLU(),
            torch.nn.Linear(D_out, D_out),
        )
        self.mean = torch.nn.Linear(H, H)
        self.var = torch.nn.Linear(H, H)

    def forward(self, x):
        encoded = self.encoder(x)
        mu = self.mean(encoded)
        logvar = 0.5 * torch.exp(self.var(encoded))
        latent = mu + torch.randn_like(encoded) * logvar
        decoded = self.decoder(latent)
        return decoded, mu, logvar


class VAE:
    def __init__(self, n_clusters, n_componets, random_state, max_iter=10000, tol=1e-4, lr=1e-3):
        self.n_clusters = n_clusters
        self.n_componets = n_componets
        self.random_state = random_state
        self.max_iter = max_iter
        self.tol = tol
        self.lr = lr
        self.labels_ = None
        self.model = None

    def fit(self, X):
        self.model = VAE_NN(X.shape[1], self.n_componets, X.shape[1]).to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        for t in range(self.max_iter):
            y_pred, mu, logvar = self.model(X)
            loss = _vae_loss_function(y_pred, X, mu, logvar)
            if t % 1000 == 999:
                print(t, loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def fit_transform(self, X):
        self.fit(X)
        _, mu, logvar = self.model(X)
        return mu + torch.randn_like(mu) * logvar
