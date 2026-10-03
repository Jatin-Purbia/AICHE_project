"""First-principles PEM electrolyzer thermal model (numpy + torch), RK4 and steady-state solver.

States x = [T_out_ele, T_in_ele, T_out_c] in K; inputs u1, u2 in [0,1]; disturbance I in A.
"""
import sklearn  # noqa: F401  (must be imported before torch, otherwise torch c10.dll fails to load on this Windows setup)
import numpy as np
import torch
import yaml
from pathlib import Path
from scipy.optimize import root

ROOT = Path(__file__).resolve().parents[1]


def load_params(path=None):
    with open(path or ROOT / "params.yaml") as fh:
        raw = yaml.safe_load(fh)
    return {k: float(v) for k, v in raw.items()}  # PyYAML reads "1.2e5" as str, so coerce


def _vcell(T, I, p, log):
    """Ulleberg-type cell voltage, theta = T - 273.15 (degC), A in m^2:
    V = V_rev + (r1 + r2*theta)/A*I + (s0 + s1*theta + s2*theta^2) * ln((t1 + t2/theta + t3/theta^2)/A*I + 1).
    NOTE: the source paper prints t2*T + t3*T^2 in the log argument (apparent typo);
    Ulleberg's original correlation uses t2/theta + t3/theta^2, which is used here."""
    th = T - 273.15
    ohm = (p["r1"] + p["r2"] * th) / p["A"] * I
    s = p["s0"] + p["s1"] * th + p["s2"] * th ** 2
    arg = (p["t1"] + p["t2"] / th + p["t3"] / th ** 2) / p["A"] * I + 1.0
    return p["V_rev"] + ohm + s * log(arg)


def vcell_numpy(T, I, p):
    return _vcell(np.asarray(T, float), np.asarray(I, float), p, np.log)


def vcell_torch(T, I, p):
    return _vcell(T, I, p, torch.log)



def _rhs(x1, x2, x3, u1, u2, I, p, vc):
    q_gen = p["n_cell"] * I * (vc - p["V_th"])
    mcp = p["m0"] * p["cp"]
    d1 = (q_gen - p["h"] * p["A_s"] * (x1 - p["T_amb"]) - u1 * mcp * (x1 - x2)) / p["C_th"]
    d2 = (u1 * mcp * (x1 - x2) + p["UA_ex"] * (x3 - x2)) / p["C_h"]
    d3 = (u2 * mcp * (p["T_inc"] - x3) + p["UA_ex"] * (x2 - x3)) / p["C_c"]
    return d1, d2, d3


def f_numpy(x, u1, u2, I, p):
    """dx/dt for x of shape (..., 3); u1, u2, I broadcastable to x[..., 0]."""
    x = np.asarray(x, float)
    x1, x2, x3 = x[..., 0], x[..., 1], x[..., 2]
    d = _rhs(x1, x2, x3, u1, u2, I, p, vcell_numpy(x1, I, p))
    return np.stack(np.broadcast_arrays(*d), axis=-1)


def f_torch(x, u1, u2, I, p):
    """Batched, differentiable dx/dt (same equations as f_numpy)."""
    x1, x2, x3 = x[..., 0], x[..., 1], x[..., 2]
    d = _rhs(x1, x2, x3, u1, u2, I, p, vcell_torch(x1, I, p))
    return torch.stack(torch.broadcast_tensors(*d), dim=-1)


def rk4_step(x, u1, u2, I, dt, p):
    """Classical RK4 step with inputs held constant over the step."""
    k1 = f_numpy(x, u1, u2, I, p)
    k2 = f_numpy(x + 0.5 * dt * k1, u1, u2, I, p)
    k3 = f_numpy(x + 0.5 * dt * k2, u1, u2, I, p)
    k4 = f_numpy(x + dt * k3, u1, u2, I, p)
    return x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate(x0, U, I_seq, dt, p):
    """Simulate len(I_seq) steps; U has shape (n, 2). Returns trajectory of shape (n+1, 3)."""
    n = len(I_seq)
    X = np.empty((n + 1, 3))
    X[0] = x0
    for k in range(n):
        X[k + 1] = rk4_step(X[k], U[k, 0], U[k, 1], I_seq[k], dt, p)
    return X


def steady_state_diag(u1, u2, I, p):
    """Steady state of f=0 (3000 s RK4 transient, dt=10 s, then scipy root polish) plus diagnostics.
    Returns (x_star, vcell_star, dict(residual, stable, in_range))."""
    x = np.full(3, p["T_amb"] + 5.0)
    for _ in range(300):
        x = rk4_step(x, u1, u2, I, 10.0, p)
    xs = root(lambda z: f_numpy(z, u1, u2, I, p), x, method="hybr", tol=1e-14).x
    res = float(np.linalg.norm(f_numpy(xs, u1, u2, I, p)))
    J = np.empty((3, 3))
    for j in range(3):
        e = np.zeros(3)
        e[j] = 1e-4
        J[:, j] = (f_numpy(xs + e, u1, u2, I, p) - f_numpy(xs - e, u1, u2, I, p)) / 2e-4
    diag = dict(residual=res, stable=bool(np.all(np.linalg.eigvals(J).real < 0)),
                in_range=bool(np.all((xs >= 295.0) & (xs <= 360.0))))
    return xs, float(vcell_numpy(xs[0], I, p)), diag


def steady_state(u1, u2, I, p):
    """Returns (x_star, vcell_star, ok); ok needs ||f||<1e-8, all Jacobian eigenvalues Re<0, T in [295, 360] K."""
    xs, v, d = steady_state_diag(u1, u2, I, p)
    return xs, v, bool(d["residual"] < 1e-8 and d["stable"] and d["in_range"])


def load_config(fast=False):
    """Read config.yaml; with fast=True the entries of its `fast:` section override the defaults."""
    with open(ROOT / "config.yaml") as fh:
        cfg = yaml.safe_load(fh)
    fastcfg = cfg.pop("fast")
    if fast:
        for sec, vals in fastcfg.items():
            if isinstance(vals, dict):
                cfg[sec].update(vals)
            else:
                cfg[sec] = vals
    cfg["is_fast"] = bool(fast)
    torch.set_num_threads(cfg["threads"])
    return cfg
