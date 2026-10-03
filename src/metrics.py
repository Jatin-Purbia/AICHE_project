"""Regression and rollout metrics. Errors are reported per output, never summed across outputs."""
import numpy as np


def regression_metrics(y, yhat):
    """Per-column RMSE, MAE, R2, NRMSE (RMSE / range of y) and max |error|. y, yhat: (N, k)."""
    e = yhat - y
    rmse = np.sqrt(np.mean(e ** 2, axis=0))
    ss_res, ss_tot = np.sum(e ** 2, axis=0), np.sum((y - y.mean(axis=0)) ** 2, axis=0)
    return dict(rmse=rmse, mae=np.mean(np.abs(e), axis=0), r2=1 - ss_res / ss_tot,
                nrmse=rmse / np.ptp(y, axis=0), maxerr=np.max(np.abs(e), axis=0))


def r2(y, yhat):
    """Per-column coefficient of determination."""
    return 1 - np.sum((yhat - y) ** 2, axis=0) / np.sum((y - y.mean(axis=0)) ** 2, axis=0)


def rollout_curve(Xtrue, Xpred):
    """Cumulative pooled RMSE over steps 1..h, h=1..T (K); pooled over trajectories and the three temperatures.
    Xtrue, Xpred: (n, T+1, 3)."""
    mse_t = np.mean((Xpred[:, 1:] - Xtrue[:, 1:]) ** 2, axis=(0, 2))
    return np.sqrt(np.cumsum(mse_t) / np.arange(1, len(mse_t) + 1))


def rollout_metrics(Xtrue, Xpred, horizons):
    """Per-state RMSE over the full horizon, pooled RMSE at each horizon H, and the pooled-RMSE curve."""
    err2 = (Xpred[:, 1:] - Xtrue[:, 1:]) ** 2
    curve = rollout_curve(Xtrue, Xpred)
    return dict(full_per_state=np.sqrt(err2.mean(axis=(0, 1))),
                at_horizon={int(h): float(curve[int(h) - 1]) for h in horizons},
                curve=curve, max_abs_err=float(np.sqrt(err2).max()))


def rollout_inst(Xtrue, Xpred):
    """Instantaneous pooled RMSE at each step t=1..T (K), pooled over trajectories and the three temperatures."""
    return np.sqrt(np.mean((Xpred[:, 1:] - Xtrue[:, 1:]) ** 2, axis=(0, 2)))
