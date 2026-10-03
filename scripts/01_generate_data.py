"""Step 1: generate steady-state and dynamic datasets, EDA figures and the dataset summary."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd

import process as pr
import datagen, preprocess, plotting

ROOT = pr.ROOT


def main(fast):
    cfg, p = pr.load_config(fast), pr.load_params()
    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)
    t0 = time.time()
    ss = datagen.gen_steady(cfg, p)
    print("steady state:", ss["log"], f"({time.time() - t0:.0f} s)")
    np.savez(ROOT / "data" / "ss.npz", inputs=ss["inputs"], outputs=ss["outputs"], keep=ss["keep"], reason=ss["reason"])
    json.dump(ss["log"], open(ROOT / "data" / "ss_log.json", "w"), indent=2)
    k = ss["keep"]
    pd.DataFrame(np.column_stack([ss["inputs"][k], ss["outputs"][k]]),
                 columns=preprocess.SS_IN + preprocess.SS_OUT).to_csv(ROOT / "data" / "ss.csv", index=False)

    t0 = time.time()
    dyn = datagen.gen_dynamic(cfg, p)
    print(f"dynamic: {len(dyn['X'])} trajectories ({time.time() - t0:.0f} s)")
    sp = dyn["split"]
    np.savez(ROOT / "data" / "dyn.npz", X=dyn["X"], U=dyn["U"], I=dyn["I"], X_test_noisy=dyn["X_test_noisy"],
             train=sp["train"], val=sp["val"], test=sp["test"])

    # EDA (variable-selection evidence) and summary
    X, Y = ss["inputs"][k], ss["outputs"][k]
    plotting.fig_ss_sampling(ss["inputs"], k)
    plotting.fig_ss_outputs_hist(Y)
    C = plotting.fig_ss_corr(X, Y)
    names = preprocess.SS_IN + preprocess.SS_OUT + ["dT"]
    pd.DataFrame(C, index=names, columns=names).to_csv(ROOT / "results" / "ss_correlation.csv")
    j = sp["train"][0]
    plotting.fig_dyn_inputs(dyn["U"][j], dyn["I"][j])
    plotting.fig_dyn_response(dyn["X"][j])

    summ = preprocess.dataset_summary(cfg, ss, dyn, preprocess.split_ss(cfg, int(k.sum())))
    json.dump(summ, open(ROOT / "results" / "dataset_summary.json", "w"), indent=2)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main("--fast" in sys.argv)
