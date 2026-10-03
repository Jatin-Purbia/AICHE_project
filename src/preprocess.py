"""Scaling, splits, lag matrices and LSTM windows. All scalers are fit on training data only."""
import numpy as np

SS_IN = ["I", "u1", "u2"]
SS_OUT = ["T_out_ele", "T_in_ele", "T_out_c", "Vcell"]


# ---------------------------------------------------------------- steady state
def fit_ss_scaler(cfg, Y_train):
    """Inputs/temperatures use the fixed design ranges ((x-c)/h); Vcell uses the training min/max -> [-1, 1]."""
    s, r = cfg["scaling"], cfg["ranges"]
    mid = np.array([s["I_center"], np.mean(r["u1"]), np.mean(r["u2"])])
    half = np.array([s["I_half"], np.ptp(r["u1"]) / 2, np.ptp(r["u2"]) / 2])
    v = Y_train[:, 3]
    return dict(x_mid=mid, x_half=half, T_c=s["T_center"], T_h=s["T_half"],
                v_mid=float((v.max() + v.min()) / 2), v_half=float((v.max() - v.min()) / 2))


def ss_x(sc, X):
    return (X - sc["x_mid"]) / sc["x_half"]


def ss_y(sc, Y):
    out = np.empty(Y.shape, dtype=float)
    out[:, :3] = (Y[:, :3] - sc["T_c"]) / sc["T_h"]
    out[:, 3] = (Y[:, 3] - sc["v_mid"]) / sc["v_half"]
    return out


def ss_y_inv(sc, Ys):
    out = np.empty(Ys.shape, dtype=float)
    out[:, :3] = Ys[:, :3] * sc["T_h"] + sc["T_c"]
    out[:, 3] = Ys[:, 3] * sc["v_half"] + sc["v_mid"]
    return out


def split_ss(cfg, n):
    """Random 70/15/15 split with the config seed. Returns dict of index arrays."""
    perm = np.random.default_rng(cfg["seed"]).permutation(n)
    f = cfg["ss"]["split"]
    a, b = int(f[0] * n), int((f[0] + f[1]) * n)
    return dict(train=perm[:a], val=perm[a:b], test=perm[b:])


# ---------------------------------------------------------------- dynamic
def fit_dyn_scaler(cfg, X_train):
    """Fixed design-range scaling for states/inputs; sigma_d = per-state std of one-step increments (train only)."""
    s, r = cfg["scaling"], cfg["ranges"]
    mid = np.array([s["I_center"], np.mean(r["u1"]), np.mean(r["u2"])])
    half = np.array([s["I_half"], np.ptp(r["u1"]) / 2, np.ptp(r["u2"]) / 2])
    dx = np.diff(X_train, axis=1).reshape(-1, 3)
    return dict(x_mid=mid, x_half=half, T_c=s["T_center"], T_h=s["T_half"], sigma_d=dx.std(axis=0))


def scale_traj(sc, X, U, I):
    """Scale trajectories: states (n,T+1,3), valves (n,T,2), current (n,T)."""
    return (X - sc["T_c"]) / sc["T_h"], (U - sc["x_mid"][1:]) / sc["x_half"][1:], (I - sc["x_mid"][0]) / sc["x_half"][0]


def increments(sc, X):
    """Scaled increment targets d_k = (x_{k+1} - x_k) / sigma_d, shape (n, T, 3)."""
    return np.diff(X, axis=1) / sc["sigma_d"]


def undo_increment(sc, x_k, d_scaled):
    """x_{k+1} = x_k + sigma_d * d_k (inverse of `increments`)."""
    return x_k + d_scaled * sc["sigma_d"]


def prev(a):
    """Shift one step back along the time axis (axis 1), replicating the first sample (edge padding)."""
    return np.concatenate([a[:, :1], a[:, :-1]], axis=1)


def lag_matrix(Xs, Us, Is):
    """Regressors [x_k, x_{k-1}, u_k, u_{k-1}, I_k, I_{k-1}] for k=0..T-1 (scaled), shape (n, T, 12).
    k=0 uses edge padding for the k-1 terms."""
    x = Xs[:, :-1]
    return np.concatenate([x, prev(x), Us, prev(Us), Is[..., None], prev(Is)[..., None]], axis=2)


def lstm_windows(Xs, Us, Is, w):
    """Windows of length w (stride 1) ending at k, features [x(3), u(2), I] scaled, edge-padded at the start.
    Returns (n, T, w, 6)."""
    F = np.concatenate([Xs[:, :-1], Us, Is[..., None]], axis=2)
    n, T, _ = F.shape
    pad = np.concatenate([np.repeat(F[:, :1], w - 1, axis=1), F], axis=1)
    idx = np.arange(T)[:, None] + np.arange(w)[None, :]
    return pad[:, idx]


def dataset_summary(cfg, ss, dyn, ss_split):
    """Counts and ranges of both datasets (written to results/dataset_summary.json)."""
    k = ss["keep"]
    X, Y = ss["inputs"][k], ss["outputs"][k]
    sp = dyn["split"]
    T = int(dyn["U"].shape[1])
    return {
        "steady": {"generated": ss["log"]["generated"], "retained": int(k.sum()), "dropped": ss["log"]["dropped"],
                   "train": len(ss_split["train"]), "val": len(ss_split["val"]), "test": len(ss_split["test"]),
                   "input_ranges": {n: [float(X[:, i].min()), float(X[:, i].max())] for i, n in enumerate(SS_IN)},
                   "output_ranges": {n: [float(Y[:, i].min()), float(Y[:, i].max())] for i, n in enumerate(SS_OUT)}},
        "dynamic": {"trajectories": int(len(dyn["X"])), "steps_per_trajectory": T, "dt_s": cfg["dyn"]["dt"],
                    "train": len(sp["train"]), "val": len(sp["val"]), "test": len(sp["test"]),
                    "train_samples": len(sp["train"]) * T,
                    "state_ranges": {n: [float(dyn["X"][..., i].min()), float(dyn["X"][..., i].max())]
                                     for i, n in enumerate(SS_OUT[:3])}},
    }
