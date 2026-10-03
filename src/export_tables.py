"""LaTeX table bodies (a single booktabs tabular each, no float/caption) written to results/tables/."""
from pathlib import Path

import numpy as np

TAB = Path(__file__).resolve().parents[1] / "results" / "tables"
SS_MODELS = ["MLR", "Poly-ridge", "MLP", "TSK / ANFIS-type"]
DYN_MODELS = ["ARX", "NARX-MLP", "LSTM", "PINC"]


def fmt(x):
    """4 significant digits, positional notation."""
    x = float(x)
    if not np.isfinite(x):
        return "n/a" if np.isnan(x) else r"$\infty$"
    if x == 0:
        return "0"
    return np.format_float_positional(x, precision=4, unique=False, fractional=False, trim="k")


def pm(vals):
    """mean +- std over seeds; a single value is shown without std."""
    v = np.asarray(vals, float)
    return fmt(v.mean()) if v.size == 1 or v.std() == 0 else f"{fmt(v.mean())} $\\pm$ {fmt(v.std())}"


def esc(s):
    """Escape text for LaTeX unless it is already math ($...$)."""
    s = str(s)
    if s.startswith("$"):
        return s
    for a, b in (("&", r"\&"), ("_", r"\_"), ("%", r"\%"), ("#", r"\#")):
        s = s.replace(a, b)
    return s.replace(" -> ", r" $\to$ ")


def write(name, colspec, header, rows):
    TAB.mkdir(parents=True, exist_ok=True)
    lines = [rf"\begin{{tabular}}{{{colspec}}}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def dataset_summary(summ, cfg):
    s, d = summ["steady"], summ["dynamic"]
    T = d["steps_per_trajectory"]
    traj = lambda n: f"{n} traj. ({n * T})"
    rows = [["Samples generated", str(s["generated"]), traj(d["trajectories"])],
            ["Samples retained", str(s["retained"]), traj(d["trajectories"])],
            ["Training set", str(s["train"]), traj(d["train"])],
            ["Validation set", str(s["val"]), traj(d["val"])],
            ["Test set", str(s["test"]), traj(d["test"])]]
    write("dataset_summary", "lcc", ["Quantity", "Steady state", "Dynamic"], rows)


def ss_metrics(res):
    """res[model] = dict(rmse=(seeds,4), r2=(seeds,4), maxerr=(seeds,4))."""
    rows = []
    for m in SS_MODELS:
        r = res[m]
        rows.append([esc(m)] + [pm(r["rmse"][:, i]) for i in range(4)] + [pm(r["r2"].mean(axis=1)), pm(r["maxerr"][:, :3].max(axis=1))])
    write("ss_metrics", "lcccccc",
          ["Model", "RMSE $T_{out,ele}$ [K]", "RMSE $T_{in,ele}$ [K]", "RMSE $T_{out,c}$ [K]", "RMSE $V_{cell}$ [V]",
           "mean $R^2$", "max $|e_T|$ [K]"], rows)


def dyn_onestep(res):
    rows = [[esc(m)] + [pm(res[m]["rmse"][:, i]) for i in range(3)] + [pm(res[m]["r2d"])] for m in DYN_MODELS]
    write("dyn_onestep_metrics", "lcccc",
          ["Model", "RMSE $x_1$ [K]", "RMSE $x_2$ [K]", "RMSE $x_3$ [K]", "$R^2_\\Delta$"], rows)


def dyn_rollout(res, horizons):
    rows = []
    for m in DYN_MODELS:
        r = res[m]
        rows.append([esc(m)] + [pm(r["H"][h]) for h in (100, 500, 2000)] + [pm(r["full"][:, i]) for i in range(3)])
    write("dyn_rollout_metrics", "lcccccc",
          ["Model", "$H=100$ [K]", "$H=500$ [K]", "$H=2000$ [K]", "$T_{out,ele}$ full [K]", "$T_{in,ele}$ full [K]",
           "$T_{out,c}$ full [K]"], rows)


def dyn_cost(cost):
    rows = [["RK4 simulator", "--", "--", fmt(cost["RK4"]["infer"])]]
    for m in DYN_MODELS:
        c = cost[m]
        rows.append([esc(m), str(c["params"]), fmt(c["train_time"]), fmt(c["infer"])])
    write("dyn_cost", "lccc", ["Model", "Parameters", "Training time [s]", "Inference / 1000 steps [s]"], rows)


def dyn_noise(res):
    rows = [[esc(m), pm(res[m]["clean"]), pm(res[m]["noisy"])] for m in DYN_MODELS]
    write("dyn_noise_metrics", "lcc", ["Model", "Clean test [K]", "Noisy test ($\\sigma_n = 0.05$ K) [K]"], rows)


def hyperparams(hyper):
    rows = []
    for m in SS_MODELS + DYN_MODELS:
        first = True
        for k, v in hyper[m].items():
            if k in ("grid", "val_loss"):
                continue
            rows.append([esc(m) if first else "", esc(k), esc(v)])
            first = False
    write("hyperparams", "lll", ["Model", "Hyperparameter", "Selected value"], rows)
