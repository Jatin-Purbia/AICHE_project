"""Data generation: Latin-hypercube steady states and random multi-level dynamic trajectories."""
import numpy as np
from scipy.stats import qmc

from process import steady_state, steady_state_diag, simulate


def gen_steady(cfg, p):
    """LHS over (I, u1, u2). Returns dict with all sampled inputs, a keep mask, outputs and drop reasons."""
    r = cfg["ranges"]
    lo = np.array([r["I"][0], r["u1"][0], r["u2"][0]])
    hi = np.array([r["I"][1], r["u1"][1], r["u2"][1]])
    n = cfg["ss"]["n"]
    inputs = qmc.scale(qmc.LatinHypercube(d=3, seed=cfg["seed"]).random(n), lo, hi)
    outputs = np.full((n, 4), np.nan)
    reason = np.array(["kept"] * n, dtype=object)
    for i, (I, u1, u2) in enumerate(inputs):
        xs, v, d = steady_state_diag(u1, u2, I, p)
        outputs[i] = [*xs, v]
        if d["residual"] >= 1e-8:
            reason[i] = "residual>=1e-8"
        elif not d["stable"]:
            reason[i] = "unstable"
        elif not d["in_range"]:
            reason[i] = "T outside [295,360] K"
    keep = reason == "kept"
    log = {"generated": int(n), "kept": int(keep.sum()),
           "dropped": {k: int((reason == k).sum()) for k in set(reason) - {"kept"}}}
    return dict(inputs=inputs, outputs=outputs, keep=keep, reason=reason.astype(str), log=log)


def _piecewise(rng, n, hold, level_fn):
    """Piecewise-constant sequence of length n, hold times ~ U[hold], levels from level_fn()."""
    out = np.empty(n)
    k = 0
    while k < n:
        h = int(rng.integers(hold[0], hold[1] + 1))
        out[k:k + h] = level_fn()
        k += h
    return out


def gen_trajectory(rng, cfg, p, limited):
    """One trajectory: random piecewise-constant I, u1, u2; start at a random steady state + N(0, 1 K^2) noise.
    If `limited`, levels stay within +-20 % of the range around a random baseline."""
    d, rg = cfg["dyn"], cfg["ranges"]
    seqs = []
    for name in ("I", "u1", "u2"):
        lo, hi = rg[name]
        if limited:
            base, amp = rng.uniform(lo, hi), d["limited_amp"] * (hi - lo)
            fn = lambda lo=lo, hi=hi, base=base, amp=amp: float(np.clip(base + rng.uniform(-amp, amp), lo, hi))
        else:
            fn = lambda lo=lo, hi=hi: rng.uniform(lo, hi)
        seqs.append(_piecewise(rng, d["T"], d["hold"], fn))
    I, u1, u2 = seqs
    lo = [rg["I"][0], rg["u1"][0], rg["u2"][0]]
    hi = [rg["I"][1], rg["u1"][1], rg["u2"][1]]
    while True:
        i0, a0, b0 = rng.uniform(lo, hi)
        xs, _, ok = steady_state(a0, b0, i0, p)
        if ok:
            break
    x0 = xs + rng.normal(0.0, d["init_noise"], 3)
    U = np.stack([u1, u2], axis=1)
    return simulate(x0, U, I, d["dt"], p), U, I


def gen_dynamic(cfg, p):
    """All trajectories, split by trajectory, plus a noisy copy of the test states.
    Arrays: X (n, T+1, 3), U (n, T, 2), I (n, T)."""
    d = cfg["dyn"]
    rng = np.random.default_rng(cfg["seed"] + 1)
    out = [gen_trajectory(rng, cfg, p, limited=(j % 2 == 1)) for j in range(d["n_traj"])]
    X, U, I = (np.stack([o[i] for o in out]) for i in range(3))
    perm = rng.permutation(d["n_traj"])
    ntr, nva, nte = d["n_train"], d["n_val"], d["n_test"]
    split = dict(train=perm[:ntr], val=perm[ntr:ntr + nva], test=perm[ntr + nva:ntr + nva + nte])
    X_test_noisy = X[split["test"]] + rng.normal(0.0, d["noise_sigma"], X[split["test"]].shape)
    return dict(X=X, U=U, I=I, split=split, X_test_noisy=X_test_noisy)
