"""Dynamic one-step models of the 3 temperatures (dt = 1 s): ARX, NARX-MLP, LSTM and PINC.

Common interface (arrays in physical units, K / valve fraction / A):
    onestep(X, U, I)   teacher-forced prediction of x_{k+1} from the true past, shape (B, T, 3)
    rollout(X0, U, I)  free run from the true initial state, predictions fed back, shape (B, T+1, 3)
    n_params()
"""
import time

import numpy as np
import torch
from sklearn.linear_model import Ridge

import preprocess as pp
from models_ss import make_mlp
from process import f_torch

D = torch.float64


def _t(a):
    return torch.as_tensor(np.asarray(a), dtype=D)


# ------------------------------------------------------------------ ARX / NARX (lagged regressors)
class LagModel:
    """x_{k+1} = x_k + sigma_d * g(z_k),  z_k = [x_k, x_{k-1}, u_k, u_{k-1}, I_k, I_{k-1}] (scaled).
    g is linear (ARX, ridge least squares) or a tanh MLP (NARX)."""

    def __init__(self, net, sc):
        self.net, self.sc = net, sc

    def onestep(self, X, U, I):
        Xs, Us, Is = pp.scale_traj(self.sc, X, U, I)
        Z = _t(pp.lag_matrix(Xs, Us, Is))
        with torch.no_grad():
            d = self.net(Z).numpy()
        return pp.undo_increment(self.sc, X[:, :-1], d)

    def rollout(self, X0, U, I):
        sc = self.sc
        _, Us, Is = pp.scale_traj(sc, np.zeros((len(U), 1, 3)), U, I)
        Us, Is = _t(Us), _t(Is)[..., None]
        x = _t(X0)
        xprev = x
        out = [x]
        sig = _t(sc["sigma_d"])
        with torch.no_grad():
            for k in range(U.shape[1]):
                km = max(k - 1, 0)
                z = torch.cat([(x - sc["T_c"]) / sc["T_h"], (xprev - sc["T_c"]) / sc["T_h"],
                               Us[:, k], Us[:, km], Is[:, k], Is[:, km]], dim=1)
                xprev, x = x, x + sig * self.net(z)
                out.append(x)
        return torch.stack(out, 1).numpy()

    def n_params(self):
        return sum(p.numel() for p in self.net.parameters())


def fit_arx(Z, T, Zva, Tva, alphas, sc):
    """Ridge least squares on the lagged regressors; alpha chosen on validation one-step MSE."""
    Z, T, Zva, Tva = (a.reshape(-1, a.shape[-1]) for a in (Z, T, Zva, Tva))
    err = lambda a: np.mean((Ridge(alpha=a).fit(Z, T).predict(Zva) - Tva) ** 2)
    alpha = min(alphas, key=err)
    r = Ridge(alpha=alpha).fit(Z, T)
    net = torch.nn.Linear(Z.shape[1], 3).double()
    net.weight.data, net.bias.data = _t(r.coef_), _t(r.intercept_)
    return LagModel(net, sc), alpha


def make_narx(width, layers, seed, sc):
    return LagModel(make_mlp(12, 3, layers, width, seed), sc)


# ------------------------------------------------------------------ LSTM
class LSTMNet(torch.nn.Module):
    def __init__(self, hidden, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.lstm = torch.nn.LSTM(6, hidden, num_layers=1, batch_first=True)
        self.head = torch.nn.Linear(hidden, 3)
        self.double()

    def forward(self, win):
        return self.head(self.lstm(win)[0][:, -1])


class LSTMModel:
    """Window of w past [x, u1, u2, I] (scaled) -> scaled increment of the next state."""

    def __init__(self, net, sc, w):
        self.net, self.sc, self.w = net, sc, w

    def onestep(self, X, U, I):
        Xs, Us, Is = pp.scale_traj(self.sc, X, U, I)
        W = pp.lstm_windows(Xs, Us, Is, self.w)
        n, T = W.shape[:2]
        Wt = _t(W.reshape(-1, self.w, 6))
        with torch.no_grad():
            d = torch.cat([self.net(Wt[j:j + 4096]) for j in range(0, len(Wt), 4096)])
        return pp.undo_increment(self.sc, X[:, :-1], d.numpy().reshape(n, T, 3))

    def rollout(self, X0, U, I):
        sc = self.sc
        _, Us, Is = pp.scale_traj(sc, np.zeros((len(U), 1, 3)), U, I)
        Us, Is = _t(Us), _t(Is)[..., None]
        x = _t(X0)
        sig = _t(sc["sigma_d"])
        B = len(x)
        row = lambda x, k: torch.cat([(x - sc["T_c"]) / sc["T_h"], Us[:, k], Is[:, k]], dim=1)[:, None]
        win = row(x, 0).repeat(1, self.w, 1)                       # edge padding with the first sample
        out = [x]
        with torch.no_grad():
            for k in range(U.shape[1]):
                if k > 0:
                    win = torch.cat([win[:, 1:], row(x, k)], dim=1)
                x = x + sig * self.net(win)
                out.append(x)
        return torch.stack(out, 1).numpy()

    def n_params(self):
        return sum(p.numel() for p in self.net.parameters())


# ------------------------------------------------------------------ PINC
class PINC:
    """Physics-informed neural network with self-loop (PINC).  N_theta(xi), xi = [x_k, u_k, I_k, tau] (scaled), tau in [0,1]
    is the fraction of one step dt.
      hard-IC:  x_hat(tau) = x_k + tau * sigma_d * N_theta(xi)           (so x_hat(0) = x_k exactly)
      soft-IC:  x_hat(tau) = T_c + T_h * N_theta(xi), extra loss ||x_hat(0)-x_k||^2 with adaptive weights.
    Physics residual: r = ((1/dt) d x_hat / d tau - f(x_hat, u_k, I_k)) / sigma_xdot.  No simulator next states are used.
    Step map F(x_k, u_k, I_k) = x_hat(tau = 1)."""

    def __init__(self, cfg, sc, p, seed):
        c = cfg["pinc"]
        self.mode, self.sc, self.p, self.dt = c["ic_mode"], sc, p, cfg["dyn"]["dt"]
        self.net = make_mlp(7, 3, c["layers"], c["width"], seed)
        self.sig_d, self.sig_xdot = _t(sc["sigma_d"]), None

    def F(self, x, u, I, tau):
        """x (N,3) K, u (N,2), I (N,), tau (N,1) -> x_hat (N,3) K. torch, differentiable."""
        sc = self.sc
        xi = torch.cat([(x - sc["T_c"]) / sc["T_h"], (u - _t(sc["x_mid"][1:])) / _t(sc["x_half"][1:]),
                        ((I - sc["x_mid"][0]) / sc["x_half"][0])[:, None], tau], dim=1)
        out = self.net(xi)
        if self.mode == "hard":
            return x + tau * self.sig_d * out
        return sc["T_c"] + sc["T_h"] * out

    def residual(self, x, u, I, tau):
        """Normalised ODE residual (N,3) at (x_k, u_k, I_k, tau); tau is created here with grad."""
        tau = tau.clone().requires_grad_(True)
        xh = self.F(x, u, I, tau)
        dx = torch.stack([torch.autograd.grad(xh[:, i].sum(), tau, create_graph=True)[0][:, 0] for i in range(3)], 1)
        return (dx / self.dt - f_torch(xh, u[:, 0], u[:, 1], I, self.p)) / self.sig_xdot

    def losses(self, x, u, I, n_c, tau=None):
        """(total, physics, IC) losses for a batch of points with n_c collocation taus per point."""
        N = len(x)
        rep = lambda a: a.repeat_interleave(n_c, 0)
        tau = torch.rand(N * n_c, 1, dtype=D) if tau is None else tau
        l_phys = torch.mean(self.residual(rep(x), rep(u), rep(I), tau) ** 2)
        if self.mode == "hard":
            return l_phys, l_phys, torch.zeros((), dtype=D)
        l_ic = torch.mean(((self.F(x, u, I, torch.zeros(N, 1, dtype=D)) - x) / self.sc["T_h"]) ** 2)
        lp, li = l_phys.detach(), l_ic.detach()
        lam_p, lam_i = li / (lp + li + 1e-30), lp / (lp + li + 1e-30)
        return lam_p * l_phys + lam_i * l_ic, l_phys, l_ic

    def step_map(self, x, u, I):
        return self.F(x, u, I, torch.ones(len(x), 1, dtype=D))

    def onestep(self, X, U, I):
        n, T = U.shape[:2]
        with torch.no_grad():
            xk = _t(X[:, :-1].reshape(-1, 3))
            out = self.step_map(xk, _t(U.reshape(-1, 2)), _t(I.reshape(-1)))
        return out.numpy().reshape(n, T, 3)

    def rollout(self, X0, U, I):
        x, Ut, It = _t(X0), _t(U), _t(I)
        out = [x]
        with torch.no_grad():
            for k in range(U.shape[1]):
                x = self.step_map(x, Ut[:, k], It[:, k])
                out.append(x)
        return torch.stack(out, 1).numpy()

    def mean_sq_residual(self, X, U, I, n_tau=8, seed=0):
        """Mean squared normalised ODE residual (and RMS in K/s) of the learned map on given states."""
        g = torch.Generator().manual_seed(seed)
        x, u, i = _t(X[:, :-1].reshape(-1, 3)), _t(U.reshape(-1, 2)), _t(I.reshape(-1))
        tot, tot_phys, cnt = 0.0, 0.0, 0
        for s in range(0, len(x), 2048):
            xb, ub, ib = (a[s:s + 2048].repeat_interleave(n_tau, 0) for a in (x, u, i))
            r = self.residual(xb, ub, ib, torch.rand(len(xb), 1, dtype=D, generator=g)).detach()
            tot += (r ** 2).sum().item()
            tot_phys += ((r * self.sig_xdot) ** 2).sum().item()
            cnt += r.numel()
        return tot / cnt, float(np.sqrt(tot_phys / cnt))

    def n_params(self):
        return sum(p.numel() for p in self.net.parameters())


def training_points(cfg, p, X, U, I, seed):
    """PINC training states: all x_k, u_k, I_k of the training trajectories plus random_frac of uniform box points
    (T in box_T, inputs in range). Returns tensors (x, u, I)."""
    rng = np.random.default_rng(seed)
    x, u, i = X[:, :-1].reshape(-1, 3), U.reshape(-1, 2), I.reshape(-1)
    n_rand = int(cfg["pinc"]["random_frac"] / (1 - cfg["pinc"]["random_frac"]) * len(x))
    r = cfg["ranges"]
    xr = rng.uniform(*cfg["pinc"]["box_T"], size=(n_rand, 3))
    ur = np.column_stack([rng.uniform(*r["u1"], n_rand), rng.uniform(*r["u2"], n_rand)])
    ir = rng.uniform(*r["I"], n_rand)
    return _t(np.vstack([x, xr])), _t(np.vstack([u, ur])), _t(np.concatenate([i, ir]))


def train_pinc(model, cfg, x, u, I):
    """Stage 1: Adam (lr 1e-3 -> 1e-4 exponential decay, mini-batches, fresh random taus each iteration);
    stage 2: L-BFGS on a fixed batch with fixed taus. Returns dict of per-iteration loss histories and wall time."""
    c = cfg["pinc"]
    p = model.p
    with torch.no_grad():                                              # sigma_xdot = std of f over training states
        f = f_torch(x, u[:, 0], u[:, 1], I, p)
        model.sig_xdot = f.std(dim=0)
    g = torch.Generator().manual_seed(0)
    hist = dict(adam=dict(total=[], phys=[], ic=[]), lbfgs=dict(total=[], phys=[], ic=[]))
    t0 = time.time()
    n_it = c["adam_iters"]
    opt = torch.optim.Adam(model.net.parameters(), lr=c["adam_lr"][0])
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, (c["adam_lr"][1] / c["adam_lr"][0]) ** (1.0 / max(n_it, 1)))
    for _ in range(n_it):
        idx = torch.randint(0, len(x), (c["batch"],), generator=g)
        tot, ph, ic = model.losses(x[idx], u[idx], I[idx], c["n_colloc"])
        opt.zero_grad()
        tot.backward()
        opt.step()
        sched.step()
        for k, v in zip(("total", "phys", "ic"), (tot, ph, ic)):
            hist["adam"][k].append(v.item())
    idx = torch.randint(0, len(x), (c["lbfgs_points"],), generator=g)
    xb, ub, ib = x[idx], u[idx], I[idx]
    tau = torch.rand(len(idx) * c["n_colloc"], 1, dtype=D, generator=g)
    lb = torch.optim.LBFGS(model.net.parameters(), lr=1.0, max_iter=c["lbfgs_iters"], history_size=50,
                           line_search_fn="strong_wolfe", tolerance_grad=1e-14, tolerance_change=1e-16)

    def closure():
        lb.zero_grad()
        tot, ph, ic = model.losses(xb, ub, ib, c["n_colloc"], tau)
        tot.backward()
        for k, v in zip(("total", "phys", "ic"), (tot, ph, ic)):
            hist["lbfgs"][k].append(v.item())
        return tot

    lb.step(closure)
    hist["time"] = time.time() - t0
    return hist


# ------------------------------------------------------------------ helpers shared by scripts
def short_rollouts(X, U, I, length, n_starts=5):
    """Sub-trajectories of `length` steps starting at evenly spaced offsets (for validation scoring)."""
    T = U.shape[1]
    offs = np.linspace(0, T - length, n_starts).astype(int)
    x0 = np.concatenate([X[:, o] for o in offs])
    Us = np.concatenate([U[:, o:o + length] for o in offs])
    Is = np.concatenate([I[:, o:o + length] for o in offs])
    Xt = np.concatenate([X[:, o:o + length + 1] for o in offs])
    return x0, Us, Is, Xt


def val_rollout_rmse(model, X, U, I, length):
    """Pooled RMSE (K) of `length`-step free-run rollouts on the given (validation) trajectories."""
    x0, Us, Is, Xt = short_rollouts(X, U, I, length)
    return float(np.sqrt(np.mean((model.rollout(x0, Us, Is)[:, 1:] - Xt[:, 1:]) ** 2)))
