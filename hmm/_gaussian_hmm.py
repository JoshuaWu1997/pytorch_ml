"""
@File   :_gaussian_hmm.py
@Author :JohsuaWu1997
@Date   :30/10/2020
"""
import torch
import numpy as np
import torch.utils.data as Data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class HMMNet(torch.nn.Module):
    def __init__(self, n_components):
        super(HMMNet, self).__init__()
        self.n_components = n_components
        self.log_transient = torch.nn.Parameter(torch.rand(n_components, n_components), requires_grad=True)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        probs = self.softmax(self.log_transient)
        out = torch.matmul(x, probs)
        return out, probs


class GaussianHMM:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X, y, n_iter=5000, lr=0.0001):  # (n, d,)
        dataset = Data.TensorDataset(X, y)
        loader = Data.DataLoader(
            dataset=dataset,
            batch_size=64,
            shuffle=True
        )
        self.model = HMMNet(self.n_components).to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()
        for t in range(n_iter):
            for step, (batch_x, batch_y) in enumerate(loader):
                y_pred, _ = self.model(batch_x)
                loss = criterion(y_pred, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if t % 100 == 99:
                print(t, loss.item())

    def predict(self, X):
        y, transient = self.model(X)
        return y, transient.detach().clone()


if __name__ == '__main__':
    X = []
    y = []
    transient = np.random.rand(5, 5)
    transient /= transient.sum(axis=1)
    for j in range(10):
        pi = np.random.rand(5, )
        pi = pi / pi.sum()
        X.append(pi.tolist())
        for i in range(10):
            pi = pi @ transient + np.random.randn(5, ) * 0.1
            X.append(pi.tolist())
            y.append(pi.tolist())
        pi = pi @ transient + np.random.randn(5, ) * 0.1
        y.append(pi.tolist())
    X = torch.tensor(X, device=device)
    y = torch.tensor(y, device=device)

    clf = GaussianHMM(5)
    clf.fit(X, y)
    y_pred, transient_pred = clf.predict(y)

    mse = torch.nn.MSELoss()
    print(mse(y[1:], y_pred[:-1]))
    print(mse(torch.from_numpy(transient).to(device), transient_pred))
