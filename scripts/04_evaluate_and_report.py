"""Step 4: evaluate all models, write figures, LaTeX tables, metrics.json/csv and results_summary.md."""
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import sklearn  # noqa: F401  (load before torch, see src/process.py)
import numpy as np
import pandas as pd
import scipy
import torch
from scipy.integrate import solve_ivp

import process as pr
import preprocess as pp
import metrics as mt
import plotting as pl
import export_tables as et
import models_dyn  # noqa: F401  (needed to unpickle the saved models)
import models_ss  # noqa: F401

ROOT = pr.ROOT
SS_M, DYN_M = et.SS_MODELS, et.DYN_MODELS
OUTN = pp.SS_OUT


def cpu_name():
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                           capture_output=True, text=True, timeout=30)
        if r.stdout.strip():
            return r.stdout.strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def clean(a):
    """Divergent rollouts (nan/inf) are clipped to +-1e6 K so that metrics stay finite and visibly huge."""
    return np.nan_to_num(a, nan=1e6, posinf=1e6, neginf=-1e6)


def solver_check(X, U, I, p):
    """Max abs difference between the RK4 trajectories (dt=1 s) and solve_ivp (DOP853, rtol=atol=1e-11)."""
    worst = 0.0
    for n in range(len(X)):
        T = U.shape[1]
        x, k = X[n, 0].copy(), 0
        while k < T:
            j = k
            while j < T and np.allclose(U[n, j], U[n, k]) and I[n, j] == I[n, k]:
                j += 1
            sol = solve_ivp(lambda t, z: pr.f_numpy(z, U[n, k, 0], U[n, k, 1], I[n, k], p), [0, j - k], x,
                            method="DOP853", rtol=1e-11, atol=1e-11, t_eval=np.arange(0, j - k + 1))
            worst = max(worst, float(np.abs(sol.y.T - X[n, k:j + 1]).max()))
            x, k = sol.y[:, -1], j
    return worst


def main(fast):
    cfg, p = pr.load_config(fast), pr.load_params()
    R = ROOT / "results"
    summ = json.load(open(R / "dataset_summary.json"))
    sstr, dytr = json.load(open(R / "ss_train.json")), json.load(open(R / "dyn_train.json"))
    M = {"seeds": cfg["seeds"], "fast": fast}

    # ------------------------------------------------------------ steady state
    sp = np.load(R / "ss_preds.npz")
    Yte = sp["Y_test"]
    ss_res, rows = {}, []
    for m in SS_M:
        per = [mt.regression_metrics(Yte, P) for P in sp[m]]
        ss_res[m] = {k: np.stack([r[k] for r in per]) for k in per[0]}
        for k in ss_res[m]:
            for i, o in enumerate(OUTN):
                rows.append(dict(model=m, metric=k, output=o, mean=ss_res[m][k][:, i].mean(), std=ss_res[m][k][:, i].std()))
    pd.DataFrame(rows).to_csv(R / "ss_metrics.csv", index=False)
    M["steady"] = {m: {k: {o: [float(v[:, i].mean()), float(v[:, i].std())] for i, o in enumerate(OUTN)} for k, v in ss_res[m].items()} for m in SS_M}
    first = {m: sp[m][0] for m in SS_M}
    pl.fig_ss_parity(Yte, first, SS_M)
    pl.fig_ss_residuals(Yte, first, SS_M)
    pl.fig_ss_learning_curve(sstr["mlp_history"])
    mlp_r2 = ss_res["MLP"]["r2"].mean()
    if mlp_r2 < 0.95:
        print("WARNING: MLP steady-state mean R2 < 0.95 -> check scaling/ranges/filtering before reporting")

    # ------------------------------------------------------------ dynamic
    d = np.load(R.parent / "data" / "dyn.npz")
    te = d["test"]
    X, U, I, Xn = d["X"][te], d["U"][te], d["I"][te], d["X_test_noisy"]
    D = torch.load(R / "models" / "dyn_models.pt", weights_only=False)
    models = D["models"]
    H = cfg["dyn"]["horizons"]
    one, roll, noise, inst, preds_roll, preds_one = {}, {}, {}, {}, {}, {}
    dX_true = (X[:, 1:] - X[:, :-1]).reshape(-1, 3)
    for name in DYN_M:
        o_rmse, o_r2d, r_full, r_H, r_curve, r_inst, n_clean, n_noisy, n1_noisy = [], [], [], [], [], [], [], [], []
        for s, m in enumerate(models[name]):
            p1 = m.onestep(X, U, I)
            o_rmse.append(np.sqrt(np.mean((p1 - X[:, 1:]) ** 2, axis=(0, 1))))
            o_r2d.append(mt.r2(dX_true, (p1 - X[:, :-1]).reshape(-1, 3)))
            pr_ = clean(m.rollout(X[:, 0], U, I))
            rm = mt.rollout_metrics(X, pr_, H)
            r_full.append(rm["full_per_state"]), r_H.append(rm["at_horizon"]), r_curve.append(rm["curve"])
            r_inst.append(mt.rollout_inst(X, pr_))
            n_clean.append(float(np.sqrt(np.mean((pr_[:, 1:] - X[:, 1:]) ** 2))))
            pn = clean(m.rollout(Xn[:, 0], U, I))                        # noisy measured initial state
            n_noisy.append(float(np.sqrt(np.mean((pn[:, 1:] - X[:, 1:]) ** 2))))
            n1_noisy.append(float(np.sqrt(np.mean((m.onestep(Xn, U, I) - X[:, 1:]) ** 2))))
            if s == 0:
                preds_roll[name], preds_one[name] = pr_[0], p1[0]
        one[name] = dict(rmse=np.array(o_rmse), r2d=np.array(o_r2d).mean(axis=1), r2d_state=np.array(o_r2d))
        roll[name] = dict(full=np.array(r_full), H={h: [r[h] for r in r_H] for h in H}, curve=np.array(r_curve), inst=np.array(r_inst))
        noise[name] = dict(clean=n_clean, noisy=n_noisy, onestep_clean=[float(np.sqrt(np.mean((m.onestep(X, U, I) - X[:, 1:]) ** 2))) for m in models[name]], onestep_noisy=n1_noisy)
        print(f"{name:9s} one-step RMSE {one[name]['rmse'].mean(0)}  R2_d {one[name]['r2d'].mean():.4f}  rollout full {roll[name]['full'].mean(0)}")

    pl.fig_dyn_onestep(X[0], {m: preds_one[m] for m in DYN_M}, DYN_M)
    pl.fig_dyn_rollout(X[0], preds_roll, DYN_M)
    pl.fig_dyn_rollout_error({m: roll[m]["curve"] for m in DYN_M}, DYN_M)

    # step responses from the common nominal steady state
    sr = cfg["step_response"]
    I0, a0, b0 = sr["nominal"]
    xn, _, ok = pr.steady_state(a0, b0, I0, p)
    L = sr["length"]
    scen, step_rmse = [], {}
    for title, (I1, a1, b1) in {"step $I$: 30 $\\to$ 45 A": (sr["I_step"], a0, b0), "step $u_1$: 0.6 $\\to$ 0.9": (I0, sr["u1_step"], b0),
                               "step $u_2$: 0.6 $\\to$ 0.9": (I0, a0, sr["u2_step"])}.items():
        Us, Is = np.tile([a1, b1], (L, 1)), np.full(L, I1)
        Xs = pr.simulate(xn, Us, Is, 1.0, p)
        P = {m: clean(models[m][0].rollout(xn[None], Us[None], Is[None])[0]) for m in DYN_M}
        scen.append((title, np.arange(L + 1), Xs, P))
        step_rmse[title] = {m: np.sqrt(np.mean((P[m] - Xs) ** 2, axis=0)).tolist() for m in DYN_M}
    pl.fig_dyn_step_response(scen, DYN_M)

    # cost: parameters, training time, inference per 1000 rollout steps (median of repeats), RK4 reference
    n1000 = 1000
    x0, U1, I1 = X[:1, 0], U[:1, :n1000], I[:1, :n1000]

    def median_time(fn):
        fn()
        ts = []
        for _ in range(cfg["timing_repeats"]):
            t0 = time.perf_counter()
            fn()
            ts.append(time.perf_counter() - t0)
        return float(np.median(ts))

    cost = {"RK4": {"infer": median_time(lambda: pr.simulate(X[0, 0], U[0, :n1000], I[0, :n1000], 1.0, p))}}
    for m in DYN_M:
        cost[m] = dict(params=int(dytr["n_params"][m]), train_time=float(dytr["time"][m]),
                       infer=median_time(lambda m=m: models[m][0].rollout(x0, U1, I1)))

    # PINC consistency and solver verification
    pinc_res = [mm.mean_sq_residual(X, U, I) for mm in models["PINC"]]
    verif = solver_check(X, U, I, p)
    print("RK4 vs solve_ivp max abs diff on test trajectories: %.3e K" % verif)

    # PINC loss figure (seed 0)
    pl.fig_pinc_loss(dytr["pinc_hist"][0])

    # ------------------------------------------------------------ tables
    et.dataset_summary(summ, cfg)
    et.ss_metrics(ss_res)
    et.dyn_onestep(one)
    et.dyn_rollout(roll, H)
    et.dyn_cost(cost)
    et.dyn_noise(noise)
    et.hyperparams({**sstr["hyper"], **dytr["hyper"]})

    # ------------------------------------------------------------ metrics.json / csv
    M["dyn_onestep"] = {m: dict(rmse_K_mean=one[m]["rmse"].mean(0), rmse_K_std=one[m]["rmse"].std(0), r2_delta_mean=float(one[m]["r2d"].mean()),
                                r2_delta_std=float(one[m]["r2d"].std())) for m in DYN_M}
    M["dyn_rollout"] = {m: dict(full_rmse_K_mean=roll[m]["full"].mean(0), full_rmse_K_std=roll[m]["full"].std(0),
                                horizon_rmse_K_mean={h: float(np.mean(roll[m]["H"][h])) for h in H},
                                horizon_rmse_K_std={h: float(np.std(roll[m]["H"][h])) for h in H}) for m in DYN_M}
    M["noise"], M["cost"], M["step_response_rmse_K"] = noise, cost, step_rmse
    M["pinc"] = dict(mean_sq_normalised_residual=[r[0] for r in pinc_res], rms_residual_K_per_s=[r[1] for r in pinc_res],
                     final_train_physics_loss=[f["phys"] for f in dytr["pinc_final"]])
    M["solver_verification_max_abs_diff_K"] = verif
    json.dump(M, open(R / "metrics.json", "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
    pd.DataFrame([dict(model=m, state=i + 1, rollout_rmse_mean=roll[m]["full"][:, i].mean(), rollout_rmse_std=roll[m]["full"][:, i].std(),
                       onestep_rmse_mean=one[m]["rmse"][:, i].mean(), onestep_rmse_std=one[m]["rmse"][:, i].std())
                  for m in DYN_M for i in range(3)]).to_csv(R / "dyn_metrics.csv", index=False)

    write_summary(cfg, p, M, ss_res, one, roll, noise, cost, pinc_res, verif, sstr, dytr, summ, step_rmse, first, Yte)


def growth(inst, t1=100, t2=2000):
    """Mean instantaneous RMSE at t=100, 500, 2000 s and log-log slope between 100 and 2000 s (err ~ t^slope)."""
    mu = inst.mean(0)
    slope = np.log(max(mu[t2 - 1], 1e-12) / max(mu[t1 - 1], 1e-12)) / np.log(t2 / t1)
    return mu[99], mu[499], mu[min(1999, len(mu) - 1)], float(slope)


def write_summary(cfg, p, M, ss_res, one, roll, noise, cost, pinc_res, verif, sstr, dytr, summ, step_rmse, first, Yte):
    L = ["# Stage 1 results summary", "",
         "All numbers below were produced by `scripts/run_all.py` (mode: %s)." % ("FAST smoke test - not for the report" if M["fast"] else "full"), ""]
    L += ["## Software and hardware", f"- Python {platform.python_version()}, numpy {np.__version__}, scipy {scipy.__version__}, "
          f"scikit-learn {sklearn.__version__}, torch {torch.__version__}, pandas {pd.__version__}",
          f"- CPU: {cpu_name()} ({torch.get_num_threads()} torch threads), {platform.platform()}", ""]
    L += ["## Datasets", f"- Steady state: {summ['steady']['generated']} LHS points, {summ['steady']['retained']} retained "
          f"(dropped: {summ['steady']['dropped'] or 'none'}); split {summ['steady']['train']}/{summ['steady']['val']}/{summ['steady']['test']}.",
          f"- Dynamic: {summ['dynamic']['trajectories']} trajectories x {summ['dynamic']['steps_per_trajectory']} s; split by trajectory "
          f"{summ['dynamic']['train']}/{summ['dynamic']['val']}/{summ['dynamic']['test']}.", ""]
    best = min(et.SS_MODELS, key=lambda m: ss_res[m]["rmse"][:, :3].mean())
    L += ["## Steady-state models (test set)", ""]
    for m in et.SS_MODELS:
        r = ss_res[m]
        L.append(f"- {m}: RMSE [K,K,K,V] = " + ", ".join(et.fmt(r['rmse'][:, i].mean()) for i in range(4)) + f"; mean R2 = {et.fmt(r['r2'].mean())}; "
                 f"max |e_T| = {et.fmt(r['maxerr'][:, :3].max())} K")
    L += [f"- **Best steady-state model (lowest mean temperature RMSE): {best}.** Its NRMSE per output = "
          + ", ".join(et.fmt(ss_res[best]["nrmse"][:, i].mean()) for i in range(4)) + ".", ""]
    DYN = et.DYN_MODELS
    bestd = min(DYN, key=lambda m: roll[m]["full"].mean())
    L += ["## Dynamic models (test trajectories)", "", "One-step (teacher forced):", ""]
    for m in DYN:
        L.append(f"- {m}: RMSE [K] = " + ", ".join(et.fmt(one[m]["rmse"][:, i].mean()) for i in range(3)) + f"; R2_delta = {et.fmt(one[m]['r2d'].mean())}")
    L += ["", "Free-run rollout, 2000 s (RMSE pooled over the 3 temperatures at horizon H; per-state RMSE over the full horizon):", ""]
    for m in DYN:
        r = roll[m]
        L.append(f"- {m}: " + ", ".join(f"H={h}: {et.fmt(np.mean(r['H'][h]))} K" for h in r["H"]) + "; full per state [K] = "
                 + ", ".join(et.fmt(r["full"][:, i].mean()) for i in range(3)))
    L += [f"- **Best dynamic model by full-horizon rollout RMSE: {bestd}.**", "", "### Rollout drift behaviour",
          "Mean instantaneous pooled RMSE at t = 100, 500, 2000 s and log-log growth exponent p between 100 and 2000 s (error ~ t^p; p near 0 = bounded offset, p near 1 = linear drift, p > 1 = accelerating):", ""]
    for m in DYN:
        a, b, c, sl = growth(roll[m]["inst"])
        kind = "bounded / saturating" if sl < 0.3 else ("roughly linear drift" if sl < 1.3 else "accelerating drift")
        L.append(f"- {m}: {et.fmt(a)}, {et.fmt(b)}, {et.fmt(c)} K; p = {sl:.2f} ({kind})")
    pt = np.mean(dytr["time_all_seeds"]["PINC"])
    fin = ["%.3e" % f["phys"] for f in dytr["pinc_final"]]
    msq, rms = ["%.3e" % r[0] for r in pinc_res], ["%.3e" % r[1] for r in pinc_res]
    L += ["", "## PINC", f"- Training time per seed: {et.fmt(pt)} s (mean over seeds; all seeds: {[round(t) for t in dytr['time_all_seeds']['PINC']]}).",
          f"- Final training physics loss (normalised residual^2, last L-BFGS evaluation): {fin}.",
          f"- Mean squared normalised ODE residual of the learned map on test states: {msq} (RMS residual {rms} K/s).", f"- IC mode: {dytr['hyper']['PINC']['ic_mode']} (hard = initial condition satisfied exactly).", ""]
    rk = cost["RK4"]["infer"]
    L += ["## Cost (inference per 1000 rollout steps, median of repeats, batch of 1)", f"- RK4 simulator: {et.fmt(rk)} s"]
    for m in DYN:
        L.append(f"- {m}: {et.fmt(cost[m]['infer'])} s ({cost[m]['infer'] / rk:.2f} x RK4), parameters {cost[m]['params']}, training time {et.fmt(cost[m]['train_time'])} s")
    L += ["", "## Noise sensitivity (sigma_n = 0.05 K added to measured test states, models trained on clean data)",
          "Rollout RMSE [K] from the noisy measured initial state vs the clean truth (noise enters only through the initial state, plus the initial lag/window for the lagged models):", ""]
    for m in DYN:
        n = noise[m]
        L.append(f"- {m}: clean {et.fmt(np.mean(n['clean']))}, noisy {et.fmt(np.mean(n['noisy']))} ({np.mean(n['noisy']) / np.mean(n['clean']):.2f} x). "
                 f"One-step teacher-forced on noisy measured states (target = clean x_k+1): clean {et.fmt(np.mean(n['onestep_clean']))} K, noisy {et.fmt(np.mean(n['onestep_noisy']))} K")
    L += ["", "## Step responses (RMSE vs simulator over 500 s, seed 0, per state [K])", ""]
    for t, d_ in step_rmse.items():
        L.append(f"- {t}: " + "; ".join(f"{m}: " + "/".join(et.fmt(v) for v in d_[m]) for m in DYN))
    L += ["", "## Solver verification", f"- Max abs difference between RK4 (dt = 1 s) and solve_ivp (DOP853, rtol = atol = 1e-11) over the test trajectories: {verif:.3e} K.", ""]
    L += ["## Process parameters", "ASSUMED / CALIBRATED parameters (see params.yaml for the sources):"]
    for line in open(ROOT / "params.yaml", encoding="utf-8"):
        if ("ASSUMED" in line or "CALIBRATED" in line) and ":" in line and not line.lstrip().startswith("#"):
            L.append("- " + line.strip())
    xs, v, _ = pr.steady_state(0.7, 0.5, 40.0, p)
    L += ["", f"Calibrated nominal point (I = 40 A, u1 = 0.7, u2 = 0.5): T_out_ele = {xs[0]:.2f} K, T_in_ele = {xs[1]:.2f} K, T_out_c = {xs[2]:.2f} K, "
          f"dT = {xs[0] - xs[1]:.2f} K, Vcell = {v:.4f} V.", ""]
    (ROOT / "results" / "results_summary.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main("--fast" in sys.argv)
