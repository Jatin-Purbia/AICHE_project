"""Independent re-check of the saved Stage 1 results: reload data + trained models and recompute key test metrics.

Compares with results/metrics.json (written by 04_evaluate_and_report.py) and with the first-principles simulator itself
(fresh steady states at random points). Exit code 1 if any check fails.  Usage: python scripts/06_verify.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import sklearn  # noqa: F401  (load before torch, see src/process.py)
import json

import numpy as np
import torch

import process as pr
import preprocess as pp
import models_ss  # noqa: F401  (needed to unpickle the saved models)
import models_dyn  # noqa: F401
import metrics as mt

ROOT = pr.ROOT
fails = []


def check(name, ok, detail):
    print(("PASS  " if ok else "FAIL  ") + f"{name}: {detail}")
    if not ok:
        fails.append(name)


def main():
    cfg, p = pr.load_config(), pr.load_params()
    M = json.load(open(ROOT / "results" / "metrics.json"))

    # 1. steady-state MLP: recompute test RMSE from the saved models and data
    d = np.load(ROOT / "data" / "ss.npz")
    k = d["keep"]
    X, Y = d["inputs"][k], d["outputs"][k]
    sv = torch.load(ROOT / "results" / "models" / "ss_models.pt", weights_only=False)
    sc, te = sv["scaler"], sv["split"]["test"]
    rm = []
    for m in sv["models"]["MLP"]:
        yh = pp.ss_y_inv(sc, m.predict(pp.ss_x(sc, X[te])))
        rm.append(mt.regression_metrics(Y[te], yh)["rmse"])
    rm = np.mean(rm, axis=0)
    ref = [M["steady"]["MLP"]["rmse"][n][0] for n in pp.SS_OUT]
    check("steady MLP test RMSE matches metrics.json", np.allclose(rm, ref, rtol=1e-6), f"{np.round(rm, 5)} vs {np.round(ref, 5)}")
    check("steady MLP accurate (temperature RMSE < 0.1 K)", bool(np.all(rm[:3] < 0.1)), f"max {rm[:3].max():.3f} K")

    # 2. surrogate against fresh simulator solves at points never used in training
    rng = np.random.default_rng(12345)
    r = cfg["ranges"]
    Z = np.column_stack([rng.uniform(*r["I"], 40), rng.uniform(*r["u1"], 40), rng.uniform(*r["u2"], 40)])
    errs = []
    for z in Z:
        xs, v, ok = pr.steady_state(z[1], z[2], z[0], p)
        if ok:
            yh = pp.ss_y_inv(sc, np.mean([m.predict(pp.ss_x(sc, z[None])) for m in sv["models"]["MLP"]], axis=0))[0]
            errs.append(np.abs(yh[:3] - xs).max())
    check("steady MLP vs fresh simulator solves (40 new points)", max(errs) < 0.5, f"max |error| {max(errs):.3f} K over {len(errs)} converged points")

    # 3. dynamic NARX: recompute the 2000 s free-run on the test trajectories
    dd = np.load(ROOT / "data" / "dyn.npz")
    Xd, Ud, Id, te = dd["X"], dd["U"], dd["I"], dd["test"]
    dv = torch.load(ROOT / "results" / "models" / "dyn_models.pt", weights_only=False)
    full = []
    for m in dv["models"]["NARX-MLP"]:
        Xp = m.rollout(Xd[te, 0], Ud[te], Id[te])
        full.append(mt.rollout_metrics(Xd[te], Xp, cfg["dyn"]["horizons"])["full_per_state"])
    full = np.mean(full, axis=0)
    ref = np.array(M["dyn_rollout"]["NARX-MLP"]["full_rmse_K_mean"])
    check("NARX rollout RMSE matches metrics.json", np.allclose(full, ref, rtol=1e-4), f"{np.round(full, 5)} vs {np.round(ref, 5)}")
    check("NARX rollout accurate (RMSE < 0.05 K over 2000 s)", bool(np.all(full < 0.05)), f"max {full.max():.4f} K")

    # 4. simulator self-consistency
    check("RK4 vs DOP853 (stored)", M["solver_verification_max_abs_diff_K"] < 1e-6, f"{M['solver_verification_max_abs_diff_K']:.2e} K")
    xs, v, ok = pr.steady_state(0.7, 0.5, 40.0, p)
    check("calibrated nominal point (T_in_ele = 315 K, dT = 5 K)", ok and abs(xs[1] - 315.0) < 0.05 and abs(xs[0] - xs[1] - 5.0) < 0.05,
          f"T_in {xs[1]:.2f} K, dT {xs[0] - xs[1]:.2f} K")

    print("\nALL CHECKS PASSED" if not fails else f"\n{len(fails)} CHECK(S) FAILED: {fails}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
