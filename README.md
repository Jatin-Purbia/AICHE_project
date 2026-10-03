# PEM electrolyzer thermal system: Stage 1 (AI modeling)

Simulates a PEM electrolyzer thermal model, generates steady-state and dynamic datasets, trains and compares AI
models, and writes figures (`figures/`), LaTeX table bodies (`results/tables/`) and metrics (`results/`).
All data are simulated.

## Install

Python 3.10+, CPU only.

```
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```
python scripts/run_all.py          # full pipeline, about 45-50 min on a laptop CPU
python scripts/run_all.py --fast   # smoke test, a few minutes (numbers not for the report)
```

Scripts can also be run one by one, in order: `scripts/01_generate_data.py`, `02_train_steady.py`,
`03_train_dynamic.py`, `04_evaluate_and_report.py` (each accepts `--fast`). `scripts/05_make_diagrams.py` draws the three block
diagrams (schematic, workflow, PINC); `run_all.py` does not call it.

## Configuration

- `config.yaml`: sizes, ranges, seeds, hyperparameter grids (the `fast:` block overrides it for `--fast`).
- `params.yaml`: process parameters, each tagged PAPER / ASSUMED / CALIBRATED. The Ulleberg cell-voltage constants
  are an unverified alkaline set; replace them if you have a PEM set.

## Outputs

- `results/results_summary.md`: plain-language summary of all numbers
- `results/metrics.json`, `results/*.csv`, `results/models/`
- `results/tables/*.tex`, `figures/*.png`

## Report

`report.tex` (repo root) reads `figures/` and `results/tables/` directly. Compile with `pdflatex report.tex` twice.

## Note (Windows)

`sklearn` must be imported before `torch`, otherwise torch fails to load `c10.dll`. The scripts and tests already
do this; keep that order in any new entry point.
