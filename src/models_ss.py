"""Steady-state models, inputs (I, u1, u2) scaled to [-1, 1], outputs 4 scaled values.
Every model has predict(Xs) -> Ys (numpy, scaled) and n_params()."""
import itertools
import time

import numpy as np
import torch
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures


def make_mlp(n_in, n_out, layers, width, seed):
    """tanh MLP with `layers` hidden layers of `width` units."""
    torch.manual_seed(seed)
    mods, d = [], n_in
    for _ in range(layers):
        mods += [torch.nn.Linear(d, width), torch.nn.Tanh()]
        d = width
    return torch.nn.Sequential(*mods, torch.nn.Linear(d, n_out)).double()


def train_net(net, Xtr, Ytr, Xva, Yva, lr, max_epochs, patience, batch, seed, clip=None):
    """Adam on MSE with early stopping on validation loss; restores the best weights.
    `clip` = gradient-norm clipping value (None = off).
    Returns dict(train, val, best_epoch, time) with per-epoch loss history."""
    Xtr, Ytr, Xva, Yva = (torch.as_tensor(a, dtype=torch.float64) for a in (Xtr, Ytr, Xva, Yva))
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    best, best_state, bad, hist = np.inf, None, 0, dict(train=[], val=[])
    t0 = time.time()
    for ep in range(max_epochs):
        net.train()
        perm = torch.randperm(len(Xtr), generator=g)
        tot = 0.0
        for i in range(0, len(Xtr), batch):
            idx = perm[i:i + batch]
            loss = torch.mean((net(Xtr[idx]) - Ytr[idx]) ** 2)
            opt.zero_grad()
            loss.backward()
            if clip:
                torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
            opt.step()
            tot += loss.item() * len(idx)
        with torch.no_grad():
            v = torch.mean((net(Xva) - Yva) ** 2).item()
        hist["train"].append(tot / len(Xtr))
        hist["val"].append(v)
        if v < best - 1e-12:
            best, bad, best_epoch = v, 0, ep
            best_state = {k: t.clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    hist.update(best_epoch=best_epoch, time=time.time() - t0)
    return hist


class SkModel:
    """Wrapper for MLR (degree=1, plain least squares) or degree-2 polynomial + ridge."""

    def __init__(self, degree=1, alpha=None):
        self.poly = PolynomialFeatures(degree, include_bias=False)
        self.reg = LinearRegression() if alpha is None else Ridge(alpha=alpha)

    def fit(self, X, Y):
        self.reg.fit(self.poly.fit_transform(X), Y)
        return self

    def predict(self, X):
        return self.reg.predict(self.poly.transform(X))

    def n_params(self):
        return int(self.reg.coef_.size + self.reg.intercept_.size)


def fit_poly_ridge(Xtr, Ytr, Xva, Yva, alphas):
    """Degree-2 polynomial + Ridge, alpha picked on validation MSE."""
    best = min((np.mean((SkModel(2, a).fit(Xtr, Ytr).predict(Xva) - Yva) ** 2), a) for a in alphas)
    return SkModel(2, best[1]).fit(Xtr, Ytr), best[1]


class MLPModel:
    def __init__(self, net):
        self.net = net

    def predict(self, X):
        with torch.no_grad():
            return self.net(torch.as_tensor(X, dtype=torch.float64)).numpy()

    def n_params(self):
        return sum(p.numel() for p in self.net.parameters())


def fit_mlp_grid(cfg, Xtr, Ytr, Xva, Yva, seed):
    """Grid over layers/width/lr, picked on best validation loss (seed `seed`). Returns (best_cfg, table)."""
    m = cfg["ss"]["mlp"]
    table = []
    for L, W, lr in itertools.product(m["layers"], m["width"], m["lr"]):
        net = make_mlp(Xtr.shape[1], Ytr.shape[1], L, W, seed)
        h = train_net(net, Xtr, Ytr, Xva, Yva, lr, m["max_epochs"], m["patience"], m["batch"], seed)
        table.append(dict(layers=L, width=W, lr=lr, val_loss=float(min(h["val"]))))
    return min(table, key=lambda r: r["val_loss"]), table


class TSK:
    """First-order Takagi-Sugeno-Kang (ANFIS-type) model with Gaussian membership functions.

    mu_ij(x_i) = exp(-0.5 ((x_i - c_ij) / s_ij)^2),  i = input, j = MF index
    rule r (one MF per input, all combinations: n_mf^3 rules):  w_r(x) = prod_i mu_{i, j_r(i)}(x_i)
    normalised firing strength:                                  wbar_r = w_r / sum_r w_r
    first-order consequent of rule r, output k:                  y_rk = a_rk0 + a_rk . x
    output:                                                       y_k = sum_r wbar_r y_rk
    Hybrid learning: for fixed premises (c, s) the consequents a solve a linear least-squares problem;
    for fixed consequents, c and log s are updated by Adam through this differentiable torch graph. The two
    steps alternate; the iterate with the lowest validation MSE is kept."""

    def __init__(self, n_in=3, n_mf=2, n_out=4):
        self.n_in, self.n_out = n_in, n_out
        self.rules = torch.tensor(list(itertools.product(range(n_mf), repeat=n_in)))        # (R, n_in)
        self.c = torch.linspace(-1, 1, n_mf, dtype=torch.float64).repeat(n_in, 1).clone().requires_grad_(True)
        self.log_s = torch.full((n_in, n_mf), np.log(0.8), dtype=torch.float64, requires_grad=True)
        self.theta = None

    def _wbar(self, X):
        mu = torch.exp(-0.5 * ((X[:, :, None] - self.c) / self.log_s.exp()) ** 2)          # (N, n_in, n_mf)
        w = torch.ones(len(X), len(self.rules), dtype=torch.float64)
        for i in range(self.n_in):
            w = w * mu[:, i, self.rules[:, i]]
        return w / (w.sum(1, keepdim=True) + 1e-12)

    def _design(self, X):
        X1 = torch.cat([torch.ones(len(X), 1, dtype=torch.float64), X], 1)
        return (self._wbar(X)[:, :, None] * X1[:, None, :]).reshape(len(X), -1)

    def _solve_consequents(self, X, Y):
        with torch.no_grad():
            Phi = self._design(X)
            self.theta = torch.linalg.solve(Phi.T @ Phi + 1e-8 * torch.eye(Phi.shape[1], dtype=torch.float64), Phi.T @ Y)

    def fit(self, Xtr, Ytr, Xva, Yva, outer_iters=5, adam_steps=200, adam_lr=0.02):
        Xtr, Ytr, Xva, Yva = (torch.as_tensor(a, dtype=torch.float64) for a in (Xtr, Ytr, Xva, Yva))
        opt = torch.optim.Adam([self.c, self.log_s], lr=adam_lr)
        best = (np.inf, None)
        for _ in range(outer_iters):
            self._solve_consequents(Xtr, Ytr)
            for _ in range(adam_steps):
                loss = torch.mean((self._design(Xtr) @ self.theta - Ytr) ** 2)
                opt.zero_grad()
                loss.backward()
                opt.step()
            self._solve_consequents(Xtr, Ytr)
            with torch.no_grad():
                v = torch.mean((self._design(Xva) @ self.theta - Yva) ** 2).item()
            if v < best[0]:
                best = (v, (self.c.detach().clone(), self.log_s.detach().clone(), self.theta.clone()))
        self.c.data, self.log_s.data, self.theta = best[1]
        return self

    def predict(self, X):
        with torch.no_grad():
            return (self._design(torch.as_tensor(X, dtype=torch.float64)) @ self.theta).numpy()

    def n_params(self):
        return self.c.numel() + self.log_s.numel() + self.theta.numel()
