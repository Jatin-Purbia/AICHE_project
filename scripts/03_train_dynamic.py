"""Step 3: train ARX, NARX-MLP, LSTM and PINC one-step models; save models, histories and hyperparameters."""
import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import sklearn  # noqa: F401  (load before torch, see src/process.py)
import numpy as np
import torch

import process as pr
import preprocess as pp
import models_ss as ms
import models_dyn as md

ROOT = pr.ROOT


def main(fast):
    cfg, p = pr.load_config(fast), pr.load_params()
    d, dc = np.load(ROOT / "data" / "dyn.npz"), None
    X, U, I = d["X"], d["U"], d["I"]
    tr, va = d["train"], d["val"]
    dc = cfg["dyn"]
    sc = pp.fit_dyn_scaler(cfg, X[tr])
    print("sigma_d [K/step]:", sc["sigma_d"])

    def prep(idx):
        Xs, Us, Is = pp.scale_traj(sc, X[idx], U[idx], I[idx])
        return pp.lag_matrix(Xs, Us, Is), pp.increments(sc, X[idx]), pp.lstm_windows(Xs, Us, Is, dc["window"])

    Ztr, Ttr, Wtr = prep(tr)
    Zva, Tva, Wva = prep(va)
    flat = lambda a: a.reshape(-1, *a.shape[2:])
    out = dict(hyper={}, time={}, n_params={}, grids={}, history={})
    models = {}
    val_len = dc["val_rollout_len"]
    score = lambda m: md.val_rollout_rmse(m, X[va], U[va], I[va], val_len)

    # ARX
    t0 = time.time()
    arx, alpha = md.fit_arx(Ztr, Ttr, Zva, Tva, dc["arx_alphas"], sc)
    out["time"]["ARX"], out["n_params"]["ARX"] = time.time() - t0, arx.n_params()
    out["hyper"]["ARX"] = {"regressors": "$[x_k, x_{k-1}, u_k, u_{k-1}, I_k, I_{k-1}]$", "ridge alpha": alpha}
    models["ARX"] = [arx]
    print("ARX done, val 100-step rollout RMSE %.4f K" % score(arx))

    # NARX-MLP: grid on width/lr, scored by validation rollout RMSE
    n = dc["narx"]
    grid = []
    for W, lr in itertools.product(n["width"], n["lr"]):
        net = md.make_narx(W, n["layers"], cfg["seed"], sc)
        ms.train_net(net.net, flat(Ztr), flat(Ttr), flat(Zva), flat(Tva), lr, n["max_epochs"], n["patience"], n["batch"], cfg["seed"])
        grid.append(dict(width=W, lr=lr, val_rollout_rmse=score(net)))
        print("NARX grid", grid[-1])
    best = min(grid, key=lambda r: r["val_rollout_rmse"])
    out["grids"]["NARX-MLP"] = grid
    models["NARX-MLP"], times = [], []
    for s in cfg["seeds"]:
        m = md.make_narx(best["width"], n["layers"], s, sc)
        h = ms.train_net(m.net, flat(Ztr), flat(Ttr), flat(Zva), flat(Tva), best["lr"], n["max_epochs"], n["patience"], n["batch"], s)
        models["NARX-MLP"].append(m), times.append(h["time"])
        out["history"].setdefault("NARX-MLP", {"train": h["train"], "val": h["val"]})
    out["hyper"]["NARX-MLP"] = dict(hidden_layers=n["layers"], width=best["width"], lr=best["lr"], activation="tanh",
                                   early_stopping_patience=n["patience"], batch=n["batch"])
    out["time"]["NARX-MLP"], out["n_params"]["NARX-MLP"] = float(np.mean(times)), models["NARX-MLP"][0].n_params()

    # LSTM
    l = dc["lstm"]
    grid = []
    for Hd, lr in itertools.product(l["hidden"], l["lr"]):
        m = md.LSTMModel(md.LSTMNet(Hd, cfg["seed"]), sc, dc["window"])
        ms.train_net(m.net, flat(Wtr), flat(Ttr), flat(Wva), flat(Tva), lr, l["max_epochs"], l["patience"], l["batch"], cfg["seed"], clip=1.0)
        grid.append(dict(hidden=Hd, lr=lr, val_rollout_rmse=score(m)))
        print("LSTM grid", grid[-1])
    best = min(grid, key=lambda r: r["val_rollout_rmse"])
    out["grids"]["LSTM"] = grid
    models["LSTM"], times = [], []
    for s in cfg["seeds"]:
        m = md.LSTMModel(md.LSTMNet(best["hidden"], s), sc, dc["window"])
        h = ms.train_net(m.net, flat(Wtr), flat(Ttr), flat(Wva), flat(Tva), best["lr"], l["max_epochs"], l["patience"], l["batch"], s, clip=1.0)
        models["LSTM"].append(m), times.append(h["time"])
        out["history"].setdefault("LSTM", {"train": h["train"], "val": h["val"]})
    out["hyper"]["LSTM"] = dict(layers=1, hidden=best["hidden"], window=dc["window"], lr=best["lr"], grad_clip=1.0,
                               early_stopping_patience=l["patience"], batch=l["batch"])
    out["time"]["LSTM"], out["n_params"]["LSTM"] = float(np.mean(times)), models["LSTM"][0].n_params()

    # PINC (no simulator next-state targets; physics residual only)
    c = cfg["pinc"]
    models["PINC"], times, out["pinc_hist"], out["pinc_final"] = [], [], [], []
    for s in cfg["seeds"]:
        xp, up, ip = md.training_points(cfg, p, X[tr], U[tr], I[tr], s)
        m = md.PINC(cfg, sc, p, s)
        h = md.train_pinc(m, cfg, xp, up, ip)
        models["PINC"].append(m), times.append(h["time"])
        out["pinc_hist"].append({k: h[k] for k in ("adam", "lbfgs")})
        out["pinc_final"].append(dict(phys=h["lbfgs"]["phys"][-1] if h["lbfgs"]["phys"] else h["adam"]["phys"][-1],
                                      adam_end=h["adam"]["phys"][-1]))
        print(f"PINC seed {s}: {h['time']:.0f} s, final physics loss {out['pinc_final'][-1]['phys']:.3e}")
    out["hyper"]["PINC"] = dict(ic_mode=c["ic_mode"], layers=c["layers"], width=c["width"], n_colloc=c["n_colloc"],
                               adam_iters=c["adam_iters"], lbfgs_iters=c["lbfgs_iters"], lr=f"{c['adam_lr'][0]} -> {c['adam_lr'][1]}",
                               batch=c["batch"], random_point_frac=c["random_frac"])
    out["time"]["PINC"], out["n_params"]["PINC"] = float(np.mean(times)), models["PINC"][0].n_params()
    out["time_all_seeds"] = {"PINC": times}

    (ROOT / "results" / "models").mkdir(parents=True, exist_ok=True)
    torch.save(dict(models=models, scaler=sc), ROOT / "results" / "models" / "dyn_models.pt")
    json.dump(out, open(ROOT / "results" / "dyn_train.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main("--fast" in sys.argv)
