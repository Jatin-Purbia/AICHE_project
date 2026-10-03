"""All figures (matplotlib, grayscale-safe: distinct line styles and markers, 200 dpi PNG)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG = Path(__file__).resolve().parents[1] / "figures"
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "legend.fontsize": 8,
                     "axes.spines.top": False, "axes.spines.right": False})

# one style per model: (color, linestyle, marker); the simulator is always solid black
STYLE = {"Simulator": ("black", "-", None), "MLR": ("0.55", ":", "s"), "Poly-ridge": ("tab:green", "--", "^"),
         "MLP": ("tab:blue", "-.", "o"), "TSK / ANFIS-type": ("tab:red", (0, (5, 1, 1, 1)), "D"),
         "ARX": ("0.55", ":", "s"), "NARX-MLP": ("tab:green", "--", "^"), "LSTM": ("tab:blue", "-.", "o"),
         "PINC": ("tab:red", (0, (5, 1, 1, 1)), "D")}
T_NAMES = ["$T_{out,ele}$ [K]", "$T_{in,ele}$ [K]", "$T_{out,c}$ [K]"]


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _line(ax, t, y, name, markevery=None, **kw):
    c, ls, m = STYLE[name]
    ax.plot(t, y, color=c, ls=ls, marker=m, ms=3.5, markevery=markevery, lw=1.4 if name == "Simulator" else 1.2, label=name, **kw)


# ------------------------------------------------------------------ steady-state EDA
def fig_ss_sampling(inputs, keep):
    names = ["I [A]", "$u_1$ [-]", "$u_2$ [-]"]
    fig, axs = plt.subplots(3, 3, figsize=(6.5, 6.2))
    for i in range(3):
        for j in range(3):
            ax = axs[i, j]
            if i == j:
                ax.hist(inputs[:, i], bins=20, color="0.6", edgecolor="k")
            else:
                ax.scatter(inputs[keep, j], inputs[keep, i], s=4, c="0.45", label="retained")
                if (~keep).any():
                    ax.scatter(inputs[~keep, j], inputs[~keep, i], s=22, marker="x", c="k", label="dropped")
            if i == 2:
                ax.set_xlabel(names[j])
            if j == 0:
                ax.set_ylabel(names[i])
    axs[0, 1].legend(loc="upper right", markerscale=1.5)
    fig.tight_layout()
    save(fig, "fig_ss_sampling.png")


def fig_ss_outputs_hist(Y):
    dT = Y[:, 0] - Y[:, 1]
    cols = [Y[:, 0], Y[:, 1], Y[:, 2], Y[:, 3], dT]
    labs = T_NAMES + ["$V_{cell}$ [V]", "$\\Delta T = T_{out,ele}-T_{in,ele}$ [K]"]
    fig, axs = plt.subplots(2, 3, figsize=(6.5, 4.0))
    for ax, c, l in zip(axs.ravel(), cols, labs):
        ax.hist(c, bins=30, color="0.65", edgecolor="k", lw=0.5)
        ax.set_xlabel(l)
        ax.set_ylabel("count")
    axs[1, 2].axis("off")
    fig.tight_layout()
    save(fig, "fig_ss_outputs_hist.png")


def fig_ss_corr(X, Y):
    """Correlation of inputs (rows) with outputs and delta T (columns); returns the full matrix."""
    D = np.column_stack([X, Y, Y[:, 0] - Y[:, 1]])
    C = np.corrcoef(D, rowvar=False)
    sub = C[:3, 3:]
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    im = ax.imshow(sub, cmap="gray", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(5), ["$T_{out,ele}$", "$T_{in,ele}$", "$T_{out,c}$", "$V_{cell}$", "$\\Delta T$"])
    ax.set_yticks(range(3), ["$I$", "$u_1$", "$u_2$"])
    ax.grid(False)
    for i in range(3):
        for j in range(5):
            ax.text(j, i, f"{sub[i, j]:.2f}", ha="center", va="center", color="k", bbox=dict(fc="w", ec="none", alpha=0.85, pad=1.5))
    fig.colorbar(im, ax=ax, label="Pearson r")
    fig.tight_layout()
    save(fig, "fig_ss_corr.png")
    return C


# ------------------------------------------------------------------ dynamic data
def fig_dyn_inputs(U, I):
    t = np.arange(len(I))
    fig, axs = plt.subplots(3, 1, figsize=(6.5, 4.8), sharex=True)
    axs[0].step(t, I, "k-", where="post")
    axs[1].step(t, U[:, 0], "k-", where="post")
    axs[2].step(t, U[:, 1], "k--", where="post")
    for ax, l in zip(axs, ["$I$ [A]", "$u_1$ [-]", "$u_2$ [-]"]):
        ax.set_ylabel(l)
    axs[2].set_xlabel("time [s]")
    fig.tight_layout()
    save(fig, "fig_dyn_inputs.png")


def fig_dyn_response(X):
    t = np.arange(len(X))
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    for i, (ls, lab) in enumerate(zip(["-", "--", ":"], T_NAMES)):
        ax.plot(t, X[:, i], "k" if i == 0 else ("0.4" if i == 1 else "0.2"), ls=ls, label=lab, lw=1.3)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("temperature [K]")
    ax.legend(ncol=3)
    fig.tight_layout()
    save(fig, "fig_dyn_response.png")


# ------------------------------------------------------------------ steady-state results
SS_LAB = ["$T_{out,ele}$ [K]", "$T_{in,ele}$ [K]", "$T_{out,c}$ [K]", "$V_{cell}$ [V]"]


def fig_ss_parity(Y, preds, models):
    """Rows = outputs, columns = models; preds[m] is (N, 4)."""
    fig, axs = plt.subplots(4, len(models), figsize=(7.0, 6.8))
    for i in range(4):
        for j, m in enumerate(models):
            ax = axs[i, j]
            lo, hi = Y[:, i].min(), Y[:, i].max()
            ax.plot([lo, hi], [lo, hi], "k-", lw=0.8)
            ax.scatter(Y[:, i], preds[m][:, i], s=6, c="0.35", marker=STYLE[m][2] or "o", alpha=0.7)
            if i == 0:
                ax.set_title(m, fontsize=8)
            if i == 3:
                ax.set_xlabel("simulator")
            if j == 0:
                ax.set_ylabel(f"predicted\n{SS_LAB[i]}")
    fig.tight_layout()
    save(fig, "fig_ss_parity.png")


def fig_ss_residuals(Y, preds, models):
    """Top row: residual (% of output range) vs predicted value (min-max normalised); bottom row: residual histograms."""
    rng = np.ptp(Y, axis=0)
    mk = ["o", "s", "^", "x"]
    fig, axs = plt.subplots(2, len(models), figsize=(7.0, 4.6))
    for j, m in enumerate(models):
        res = (preds[m] - Y) / rng * 100
        for i in range(4):
            axs[0, j].scatter((preds[m][:, i] - Y[:, i].min()) / rng[i], res[:, i], s=5, marker=mk[i], c="0.3" if i % 2 else "0.6", label=SS_LAB[i] if j == 0 else None)
            axs[1, j].hist(res[:, i], bins=30, histtype="step", ls=["-", "--", ":", "-."][i], color="k", lw=1.0)
        axs[0, j].axhline(0, c="k", lw=0.6)
        axs[0, j].set_title(m, fontsize=8)
        axs[1, j].set_xlabel("residual [% of range]")
        axs[0, j].set_xlabel("predicted (normalised)")
    axs[0, 0].set_ylabel("residual [% of range]")
    axs[1, 0].set_ylabel("count")
    axs[0, 0].legend(fontsize=6, markerscale=1.5)
    fig.tight_layout()
    save(fig, "fig_ss_residuals.png")


def fig_ss_learning_curve(hist):
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    ax.semilogy(hist["train"], "k-", label="training")
    ax.semilogy(hist["val"], "k--", label="validation")
    ax.axvline(hist["best_epoch"], c="0.5", ls=":", label="selected epoch")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE (scaled outputs)")
    ax.legend()
    fig.tight_layout()
    save(fig, "fig_ss_learning_curve.png")


# ------------------------------------------------------------------ dynamic results
def fig_dyn_onestep(X, preds, models, window=300):
    """One test trajectory. Left: states over the first `window` s, truth vs one-step predictions. Right: one-step errors."""
    t = np.arange(len(X) - 1) + 1
    fig, axs = plt.subplots(3, 2, figsize=(7.0, 6.0))
    for i in range(3):
        _line(axs[i, 0], t[:window], X[1:window + 1, i], "Simulator")
        for m in models:
            _line(axs[i, 0], t[:window], preds[m][:window, i], m, markevery=25)
            _line(axs[i, 1], t, preds[m][:, i] - X[1:, i], m, markevery=150)
        axs[i, 0].set_ylabel(T_NAMES[i])
        axs[i, 1].set_ylabel("one-step error [K]")
    axs[2, 0].set_xlabel("time [s]")
    axs[2, 1].set_xlabel("time [s]")
    axs[0, 0].legend(ncol=2)
    fig.tight_layout()
    save(fig, "fig_dyn_onestep.png")


def fig_dyn_rollout(X, preds, models):
    t = np.arange(len(X))
    fig, axs = plt.subplots(3, 1, figsize=(6.5, 6.0), sharex=True)
    for i in range(3):
        _line(axs[i], t, X[:, i], "Simulator")
        for m in models:
            _line(axs[i], t, preds[m][:, i], m, markevery=200)
        axs[i].set_ylabel(T_NAMES[i])
    axs[2].set_xlabel("time [s]")
    axs[0].legend(ncol=3)
    fig.tight_layout()
    save(fig, "fig_dyn_rollout.png")


def fig_dyn_rollout_error(curves, models):
    """curves[m] = (seeds, T) cumulative pooled RMSE; mean line with +-std band."""
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    for m in models:
        c = curves[m]
        h = np.arange(1, c.shape[1] + 1)
        mu, sd = c.mean(0), c.std(0)
        _line(ax, h, mu, m, markevery=250)
        ax.fill_between(h, np.maximum(mu - sd, 1e-12), mu + sd, color=STYLE[m][0], alpha=0.2, lw=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("rollout horizon $H$ [s]")
    ax.set_ylabel("RMSE over steps $1..H$ [K]")
    ax.legend(ncol=2)
    fig.tight_layout()
    save(fig, "fig_dyn_rollout_error.png")


def fig_dyn_step_response(scen, models):
    """scen = list of (title, t, Xsim, {model: X}) for the three step tests."""
    fig, axs = plt.subplots(3, len(scen), figsize=(7.2, 6.2), sharex=True)
    for j, (title, t, Xs, P) in enumerate(scen):
        axs[0, j].set_title(title, fontsize=8)
        for i in range(3):
            _line(axs[i, j], t, Xs[:, i], "Simulator")
            for m in models:
                _line(axs[i, j], t, P[m][:, i], m, markevery=60)
            if j == 0:
                axs[i, j].set_ylabel(T_NAMES[i])
        axs[2, j].set_xlabel("time [s]")
    axs[0, 0].legend(fontsize=6)
    fig.tight_layout()
    save(fig, "fig_dyn_step_response.png")


def fig_pinc_loss(hist):
    """hist = dict(adam=dict(total,phys,ic), lbfgs=...). Left: Adam stage; right: L-BFGS stage (per function evaluation)."""
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.2))
    for ax, key, xl in zip(axs, ("adam", "lbfgs"), ("Adam iteration", "L-BFGS function evaluation")):
        h = hist[key]
        ax.semilogy(h["total"], "k-", lw=1.0, label="total")
        ax.semilogy(h["phys"], "k--", lw=1.0, label="physics")
        if max(h["ic"], default=0) > 0:
            ax.semilogy(h["ic"], "k:", lw=1.0, label="IC")
        ax.set_xlabel(xl)
    axs[0].set_ylabel("loss")
    axs[0].legend()
    fig.tight_layout()
    save(fig, "fig_pinc_loss.png")
