#!/usr/bin/env python3
"""
CVA-SACS v6 — One-Click Setup & Launch
========================================
Run this ONCE after extracting the zip:

    python setup_and_run.py

It will:
  1. Check Python version
  2. Create virtual environment
  3. Install all dependencies
  4. Verify imports work
  5. Launch the dashboard
"""

import subprocess, sys, os, platform

def run(cmd, desc, check=True):
    print(f"\n  → {desc}...")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"    FAILED: {result.stderr[:300]}")
        return False
    if result.stdout.strip():
        for line in result.stdout.strip().split("\n")[:5]:
            print(f"    {line}")
    return True

print("=" * 60)
print("  CVA-SACS v6 — Setup & Launch")
print("=" * 60)

# ── 1. Check Python ──────────────────────────────────────────
v = sys.version_info
print(f"\n  Python: {v.major}.{v.minor}.{v.micro}")
if v.major < 3 or (v.major == 3 and v.minor < 9):
    print("  ERROR: Python 3.9+ required. Download from python.org")
    sys.exit(1)
print("  ✓ Python version OK")

# ── 2. Check if we're in the right directory ─────────────────
if not os.path.exists("cva_sacs_v6.py"):
    print("\n  ERROR: cva_sacs_v6.py not found in current directory.")
    print("  Make sure you've extracted the zip and cd'd into the folder:")
    print("    cd Capstone/")
    print("    python setup_and_run.py")
    sys.exit(1)
print("  ✓ Project files found")

# ── 3. Create virtual environment ────────────────────────────
venv_dir = ".venv"
pip_cmd = f"{venv_dir}/bin/pip" if platform.system() != "Windows" else f"{venv_dir}\\Scripts\\pip"
python_cmd = f"{venv_dir}/bin/python" if platform.system() != "Windows" else f"{venv_dir}\\Scripts\\python"
streamlit_cmd = f"{venv_dir}/bin/streamlit" if platform.system() != "Windows" else f"{venv_dir}\\Scripts\\streamlit"

if not os.path.exists(venv_dir):
    run(f"{sys.executable} -m venv {venv_dir}", "Creating virtual environment")
else:
    print(f"\n  ✓ Virtual environment already exists ({venv_dir}/)")

# ── 4. Upgrade pip ───────────────────────────────────────────
run(f"{pip_cmd} install --upgrade pip", "Upgrading pip", check=False)

# ── 5. Install core dependencies ─────────────────────────────
print("\n  Installing dependencies (this takes 3-5 minutes)...")
print("  ─────────────────────────────────────────────────────")

# Install in stages so failures are clear
stages = [
    ("streamlit plotly pandas numpy pyarrow requests", "Core (Streamlit + data)"),
    ("scikit-learn scipy", "ML base (sklearn + scipy)"),
    ("xgboost lightgbm", "Boosting (XGBoost + LightGBM)"),
    ("catboost", "CatBoost (slow install, be patient)"),
    ("optuna", "Optuna (hyperparameter tuning)"),
    ("yfinance", "Yahoo Finance (data)"),
    ("shap", "SHAP (explainability)"),
    ("pytest", "Pytest (testing)"),
]

for pkgs, desc in stages:
    if not run(f"{pip_cmd} install {pkgs}", desc, check=False):
        print(f"    WARNING: {desc} failed — some features may be unavailable")

# Optional heavy dependencies
print("\n  Optional (skip if you don't need them):")
run(f"{pip_cmd} install transformers torch --no-cache-dir", 
    "FinBERT (440MB download — skip with Ctrl+C if slow)", check=False)
run(f"{pip_cmd} install prophet", "Prophet (time series forecasting)", check=False)

# ── 6. Verify imports ────────────────────────────────────────
print("\n  Verifying imports...")
verify_script = '''
import sys
checks = []
for mod, name in [
    ("streamlit", "Streamlit"), ("plotly", "Plotly"), ("yfinance", "yfinance"),
    ("pandas", "Pandas"), ("numpy", "NumPy"), ("sklearn", "scikit-learn"),
    ("xgboost", "XGBoost"), ("lightgbm", "LightGBM"), ("scipy", "SciPy"),
    ("shap", "SHAP"), ("optuna", "Optuna"),
]:
    try:
        __import__(mod)
        checks.append(f"    ✓ {name}")
    except ImportError:
        checks.append(f"    ✗ {name} — NOT INSTALLED")
for mod, name in [
    ("catboost", "CatBoost"), ("transformers", "FinBERT/Transformers"),
    ("prophet", "Prophet"),
]:
    try:
        __import__(mod)
        checks.append(f"    ✓ {name}")
    except ImportError:
        checks.append(f"    ○ {name} — optional, not installed")
print("\\n".join(checks))
'''
subprocess.run([python_cmd, "-c", verify_script])

# ── 7. Run tests ─────────────────────────────────────────────
print("\n  Running unit tests...")
run(f"{python_cmd} -m pytest tests/ -v --tb=line -q", "Unit tests", check=False)

# ── 8. Create launcher scripts ───────────────────────────────
if platform.system() != "Windows":
    with open("run.sh", "w") as f:
        f.write(f"#!/bin/bash\n{streamlit_cmd} run cva_sacs_v6.py\n")
    os.chmod("run.sh", 0o755)
    print("\n  Created: run.sh (./run.sh to launch)")
else:
    with open("run.bat", "w") as f:
        f.write(f"@echo off\n{streamlit_cmd} run cva_sacs_v6.py\n")
    print("\n  Created: run.bat (double-click to launch)")

# ── 9. Launch ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  SETUP COMPLETE!")
print("=" * 60)
print(f"""
  To launch the dashboard:

    {streamlit_cmd} run cva_sacs_v6.py

  Or use the shortcut:
    {'./run.sh' if platform.system() != 'Windows' else 'run.bat'}

  The dashboard will open at:
    http://localhost:8501

  Pages:
    1. MARKET OVERVIEW    — live watchlist scan
    2. STOCK SCREENER     — any tickers, ranked by CRI
    3. DEEP ANALYSIS      — full ML pipeline + FinBERT
    4. BACKTEST           — equity curve, Sharpe, Kelly
    5. COMPARISON         — CRI vs CRI, correlation
    6. PORTFOLIO RISK     — weighted CRI, bubble chart
    7. EXPLAINABILITY     — SHAP waterfall, what-if
    8. MONTE CARLO        — forward simulation
    9. CONFORMAL          — prediction sets with guarantees
""")

# Ask to launch
ans = input("  Launch now? (y/n): ").strip().lower()
if ans in ("y", "yes", ""):
    print(f"\n  Starting Streamlit...")
    os.system(f"{streamlit_cmd} run cva_sacs_v6.py")
