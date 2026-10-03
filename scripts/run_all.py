"""Run the whole Stage 1 pipeline: python scripts/run_all.py [--fast]"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
for name in ["01_generate_data.py", "02_train_steady.py", "03_train_dynamic.py", "04_evaluate_and_report.py"]:
    t0 = time.time()
    print(f"\n=== {name} ===", flush=True)
    r = subprocess.run([sys.executable, str(HERE / name)] + [a for a in sys.argv[1:] if a == "--fast"])
    if r.returncode:
        sys.exit(f"{name} failed")
    print(f"{name} finished in {time.time() - t0:.0f} s", flush=True)
