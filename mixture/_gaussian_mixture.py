"""
@File   :_gaussian_mixture.py
@Author :JohsuaWu1997
@Date   :25/10/2020
"""
import torch
import numpy as np
import torch.utils.data as Data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GMMNet(torch.nn.Module):
    def __init__(self, n_components, n_features, mu=None, var=None):
        super(GMMNet, self).__init__()
        self.n_components = n_components
        self.n_features = n_features
        if mu is None:
            self.mu = torch.nn.Parameter(torch.randn(n_components, n_features), requires_grad=True)
        else:
            self.mu = torch.nn.Parameter(mu.detach().clone(), requires_grad=True)
        if var is None:
            self.logvar = torch.nn.Parameter(torch.rand(n_components, n_features), requires_grad=True)
        else:
            self.logvar = torch.nn.Parameter(var.detach().clone(), requires_grad=True)
        self.pi = torch.nn.Parameter(torch.rand(1, n_components), requires_grad=True)
        self.log2pi = -.5 * np.log(2. * np.pi)
        self.pi_softmax = torch.nn.Softmax(dim=1)
        self.pk_softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        log_prob = -.5 * ((x.unsqueeze(1) - self.mu.unsqueeze(0)).pow(2) / self.logvar.exp() + self.logvar).sum(dim=2)
        log_prob += self.log2pi * x.shape[1] + self.pi_softmax(self.pi).log()
        log_likelihood = torch.logsumexp(log_prob, dim=1, keepdim=True).mean()
        p_k = self.pk_softmax(log_prob)
        return p_k, -log_likelihood


class GaussianMixture:
    def __init__(self, n_components, n_features, mu=None, var=None):
        super(GaussianMixture, self).__init__()
        self.n_components = n_components
        self.n_features = n_features
        self.mu = mu
        self.var = var

    def fit(self, X, delta=1e-2, eps=1.e-6, n_iter=10000, lr=0.001):  # (n, d,)
        dataset = Data.TensorDataset(X)
        loader = Data.DataLoader(
            dataset=dataset,
            batch_size=64,
            shuffle=True
        )
        self.model = GMMNet(self.n_components, self.n_features, self.mu, self.var).to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        for t in range(n_iter):
            for step, (batch_x,) in enumerate(loader):
                y_pred, loss = self.model(batch_x)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if t % 100 == 99:
                print(t, loss.item())

    def fit_predict(self, X):
        self.fit(X)
        probs, log_likelihood = self.model(X)
        return probs.detach().clone()


if __name__ == '__main__':
    data = []
    for i in range(5):
        tmp = np.random.randn(4, 5) * 0.01
        tmp[:, i] += np.random.randn(4, ) * 2
        data.extend(tmp.tolist())
    data = torch.tensor(data, device=device)
    print(data)

    clf = GaussianMixture(5, 5)
    print(clf.fit_predict(data))
