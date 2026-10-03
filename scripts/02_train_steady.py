"""Step 2: train MLR, poly-ridge, MLP (grid + 3 seeds) and TSK on the steady-state data; save models and test predictions."""
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
import metrics

ROOT = pr.ROOT


def main(fast):
    cfg = pr.load_config(fast)
    d = np.load(ROOT / "data" / "ss.npz")
    k = d["keep"]
    X, Y = d["inputs"][k], d["outputs"][k]
    sp = pp.split_ss(cfg, len(X))
    sc = pp.fit_ss_scaler(cfg, Y[sp["train"]])
    Xs, Ys = pp.ss_x(sc, X), pp.ss_y(sc, Y)
    tr, va, te = (sp[s] for s in ("train", "val", "test"))
    out = dict(hyper={}, time={}, n_params={})
    preds = {}

    t0 = time.time()
    m = ms.SkModel(1).fit(Xs[tr], Ys[tr])
    out["time"]["MLR"], out["n_params"]["MLR"] = time.time() - t0, m.n_params()
    out["hyper"]["MLR"] = {"type": "ordinary least squares"}
    models = {"MLR": [m]}

    t0 = time.time()
    m, alpha = ms.fit_poly_ridge(Xs[tr], Ys[tr], Xs[va], Ys[va], cfg["ss"]["poly_alphas"])
    out["time"]["Poly-ridge"], out["n_params"]["Poly-ridge"] = time.time() - t0, m.n_params()
    out["hyper"]["Poly-ridge"] = {"degree": 2, "alpha": alpha}
    models["Poly-ridge"] = [m]

    best, table = ms.fit_mlp_grid(cfg, Xs[tr], Ys[tr], Xs[va], Ys[va], cfg["seed"])
    mlps, hists, times = [], [], []
    for s in cfg["seeds"]:
        net = ms.make_mlp(3, 4, best["layers"], best["width"], s)
        h = ms.train_net(net, Xs[tr], Ys[tr], Xs[va], Ys[va], best["lr"], cfg["ss"]["mlp"]["max_epochs"],
                         cfg["ss"]["mlp"]["patience"], cfg["ss"]["mlp"]["batch"], s)
        mlps.append(ms.MLPModel(net)), hists.append(h), times.append(h["time"])
    models["MLP"] = mlps
    out["hyper"]["MLP"] = dict(best, activation="tanh", optimizer="Adam", early_stopping_patience=cfg["ss"]["mlp"]["patience"],
                               batch=cfg["ss"]["mlp"]["batch"], grid=table)
    out["time"]["MLP"], out["n_params"]["MLP"] = float(np.mean(times)), mlps[0].n_params()
    out["mlp_history"] = {"train": hists[0]["train"], "val": hists[0]["val"], "best_epoch": hists[0]["best_epoch"]}

    t0 = time.time()
    t = cfg["ss"]["tsk"]
    tsk = ms.TSK(3, t["n_mf"], 4).fit(Xs[tr], Ys[tr], Xs[va], Ys[va], t["outer_iters"], t["adam_steps"], t["adam_lr"])
    out["time"]["TSK / ANFIS-type"], out["n_params"]["TSK / ANFIS-type"] = time.time() - t0, tsk.n_params()
    out["hyper"]["TSK / ANFIS-type"] = dict(mfs_per_input=t["n_mf"], rules=t["n_mf"] ** 3, mf_type="Gaussian",
                                            consequents="first-order", outer_iters=t["outer_iters"],
                                            adam_steps=t["adam_steps"], adam_lr=t["adam_lr"])
    models["TSK / ANFIS-type"] = [tsk]

    arrs = dict(X_test=X[te], Y_test=Y[te])
    for name, lst in models.items():
        arrs[name] = np.stack([pp.ss_y_inv(sc, mm.predict(Xs[te])) for mm in lst])  # (n_models, N, 4)
        r2 = metrics.regression_metrics(Y[te], arrs[name][0])["r2"]
        print(f"{name:18s} test R2 per output: {np.round(r2, 5)}  params={out['n_params'][name]}")
    np.savez(ROOT / "results" / "ss_preds.npz", **arrs)
    (ROOT / "results" / "models").mkdir(parents=True, exist_ok=True)
    torch.save(dict(models=models, scaler=sc, split=sp), ROOT / "results" / "models" / "ss_models.pt")
    json.dump(out, open(ROOT / "results" / "ss_train.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main("--fast" in sys.argv)
