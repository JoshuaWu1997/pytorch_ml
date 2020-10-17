"""
@File   :t_sne.py
@Author :JohsuaWu1997
@Date   :16/10/2020
"""
import torch
import torch.utils.data as Data
import numpy as np
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TSNE_NN(torch.nn.Module):
    def __init__(self, n_samples, n_features):
        super(TSNE_NN, self).__init__()
        self.n_samples = n_samples
        self.n_features = n_features
        self.X = torch.nn.Parameter(torch.randn(n_samples, n_features), requires_grad=True)

    def forward(self, x):
        x_pdist = torch.pdist(self.X)
        x_pdist = 1 / (1 + x_pdist)
        x_probs = x_pdist / x_pdist.sum()
        return x_probs, self.X


def tsne_loss(y, x):
    x_pdist = torch.exp(-torch.pdist(x))
    x_pdist /= x_pdist.sum()
    y_pdist = 1 / (1 + torch.pdist(y))
    y_pdist /= y_pdist.sum()
    return torch.sum(y_pdist * (torch.log(y_pdist / x_pdist)))


class TSNE:
    def __init__(self, n_components=2, perplexity=30, lr=1e-3, n_iter=10000):
        self.n_components = n_components
        self.perplexity = perplexity
        self.lr = lr
        self.n_iter = n_iter
        self.model = None

    def fit(self, X):
        dataset = Data.TensorDataset(X, X)
        loader = Data.DataLoader(
            dataset=dataset,
            batch_size=64,
            shuffle=True
        )
        self.model = TSNE_NN(X.shape[0], self.n_components).to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        for t in range(self.n_iter):
            for step, (batch_x, batch_y) in enumerate(loader):
                y_pred, _ = self.model(batch_x)
                loss = tsne_loss(y_pred, batch_y)
                if t % 1000 == 999:
                    print(t, loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(t, loss.item())

    def fit_transform(self, X):
        self.fit(X)
        _, X_embedded = self.model(X)
        return X_embedded.detach().clone()


if __name__ == '__main__':
    X = torch.randn(100, 20).to(device)
    X_embedded = TSNE(n_components=2).fit_transform(X).cpu().numpy()
    import matplotlib.pyplot as plt

    plt.scatter(X_embedded[:, 0], X_embedded[:, 1])
    plt.show()
