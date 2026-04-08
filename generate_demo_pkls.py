"""
CVA-SACS v6 — Demo PKL Generator
==================================
Generates pre-trained model PKLs using synthetic (realistic) price data
so that the dashboard demo pages load instantly without needing to
train live. Run this once before the demo:

    python generate_demo_pkls.py

Produces ./models_v6/model_AAPL.pkl, model_MSFT.pkl, model_NVDA.pkl etc.
NOTE: Must be run on the SAME machine as the dashboard to avoid
      XGBoost/sklearn version mismatch warnings.
"""

import warnings
warnings.filterwarnings("ignore")
import os
os.environ["PYTHONWARNINGS"] = "ignore"

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# ── Synthetic realistic price generator ───────────────────────────

def make_synthetic_ohlcv(ticker: str, n_days: int = 1600,
                          start_price: float = 150.0,
                          annual_ret: float = 0.12,
                          annual_vol: float = 0.25,
                          seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV that looks realistic enough for demo."""
    rng = np.random.RandomState(seed)
    dt  = 1 / 252
    mu  = annual_ret * dt
    sig = annual_vol * np.sqrt(dt)

    dates = pd.bdate_range(end=datetime.today(), periods=n_days)
    log_ret = rng.normal(mu - 0.5 * sig**2, sig, n_days)
    close   = start_price * np.exp(np.cumsum(log_ret))

    # Realistic OHLV from close
    daily_range = rng.uniform(0.005, 0.025, n_days)
    high   = close * (1 + daily_range * rng.uniform(0.4, 1.0, n_days))
    low    = close * (1 - daily_range * rng.uniform(0.4, 1.0, n_days))
    open_  = close * (1 + rng.normal(0, 0.005, n_days))
    vol    = rng.lognormal(np.log(5_000_000), 0.5, n_days)

    df = pd.DataFrame({
        "Date":   dates,
        "Open":   np.abs(open_),
        "High":   np.abs(high),
        "Low":    np.abs(low),
        "Close":  np.abs(close),
        "Volume": vol.astype(float),
        "Ticker": ticker,
    })
    # Ensure High >= Close >= Low
    df["High"]  = df[["Open","High","Close"]].max(axis=1)
    df["Low"]   = df[["Open","Low","Close"]].min(axis=1)
    return df.reset_index(drop=True)


# ── Tickers to pre-generate ───────────────────────────────────────

DEMO_TICKERS = {
    "AAPL":        dict(start_price=170.0, annual_ret=0.18, annual_vol=0.28, seed=10),
    "MSFT":        dict(start_price=310.0, annual_ret=0.20, annual_vol=0.25, seed=11),
    "NVDA":        dict(start_price=450.0, annual_ret=0.55, annual_vol=0.55, seed=12),
    "JPM":         dict(start_price=175.0, annual_ret=0.14, annual_vol=0.22, seed=13),
    "RELIANCE.NS": dict(start_price=2800.0, annual_ret=0.12, annual_vol=0.24, seed=14),
    "INFY.NS":     dict(start_price=1500.0, annual_ret=0.10, annual_vol=0.26, seed=15),
    "GOOGL":       dict(start_price=140.0, annual_ret=0.16, annual_vol=0.27, seed=16),
    "TSLA":        dict(start_price=200.0, annual_ret=0.22, annual_vol=0.65, seed=17),
}


def generate_all():
    from cva_sacs_v6_ml import (
        FeatureEngineerV6, build_label_v6, EnsembleV6, save_model
    )

    print("\n" + "═"*60)
    print("  CVA-SACS v6 — Demo PKL Generator")
    print("═"*60)

    models_dir = Path("./models_v6")
    models_dir.mkdir(exist_ok=True)

    for ticker, params in DEMO_TICKERS.items():
        safe = ticker.replace(".", "_").replace("^", "")
        pkl_path = models_dir / f"model_{safe}.pkl"

        if pkl_path.exists():
            print(f"\n  {ticker}: already exists — skipping")
            continue

        print(f"\n  {ticker}: generating synthetic data + training...")

        try:
            df_raw = make_synthetic_ohlcv(ticker, n_days=1600, **params)
            print(f"    Data: {len(df_raw)} rows")

            fe   = FeatureEngineerV6()
            df_f = fe.build(df_raw, macro=None)
            df_f = build_label_v6(df_f, h1=5, h2=10)

            fcs  = fe.get_feature_cols(df_f)
            fcs  = [c for c in fcs if c in df_f.columns]
            df_f = df_f.dropna(subset=fcs + ["risk_label"]).reset_index(drop=True)

            print(f"    Features: {len(fcs)}, Rows: {len(df_f)}, "
                  f"Labels: {sorted(df_f['risk_label'].unique())}")

            ens = EnsembleV6()
            # Use fewer trees for demo speed — still produces good signals
            import xgboost as xgb
            import lightgbm as lgb
            import catboost as cb
            from sklearn.preprocessing import RobustScaler
            from sklearn.linear_model import LogisticRegression

            X = df_f[fcs].values
            y = df_f["risk_label"].values
            n = len(X); i1 = int(n*0.60); i2 = int(n*0.80)
            ens.scaler = RobustScaler()
            Xsc = ens.scaler.fit_transform(X)
            counts = np.bincount(y, minlength=5)
            n_cls  = len(np.unique(y))
            sw = np.array([len(y)/(n_cls*max(counts[c],1)) for c in y])

            xgb_m = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.08,
                objective="multi:softprob", num_class=n_cls, eval_metric="mlogloss",
                verbosity=0, n_jobs=-1, random_state=42)
            xgb_m.fit(Xsc, y, sample_weight=sw)
            ens.base_models["xgb"] = xgb_m

            lgb_m = lgb.LGBMClassifier(n_estimators=200, max_depth=4, learning_rate=0.08,
                objective="multiclass", num_class=n_cls, verbosity=-1, n_jobs=-1,
                random_state=42)
            lgb_m.fit(Xsc, y)
            ens.base_models["lgb"] = lgb_m

            # Stacking meta
            oof = np.hstack([ens._pad_proba(m.predict_proba(Xsc[i1:i2]))
                             for m in ens.base_models.values()])
            meta = LogisticRegression(C=0.5, max_iter=500, random_state=42, solver="lbfgs")
            meta.fit(oof, y[i1:i2])
            ens.meta_lr = meta
            ens.feature_cols = fcs
            ens.train_classes_ = np.unique(y)
            ens.is_fitted = True

            # Quick accuracy check
            preds, conf, _ = ens.predict_with_confidence(df_f[fcs].tail(50))
            true  = df_f["risk_label"].values[-50:]
            acc   = float((preds == true).mean())
            print(f"    In-sample acc (last 50): {acc:.3f}")

            save_model(ticker, ens, fcs, {
                "source":   "synthetic_demo",
                "n_rows":   len(df_f),
                "n_features": len(fcs),
                "in_sample_acc": round(acc, 3),
                "generated_at":  datetime.now().isoformat(),
            })
            print(f"    ✓ Saved: model_{safe}.pkl")

        except Exception as e:
            print(f"    ✗ Error for {ticker}: {e}")
            import traceback; traceback.print_exc()

    print(f"\n{'═'*60}")
    print("  Done. PKLs ready in ./models_v6/")
    print("  Launch dashboard: streamlit run cva_sacs_v6.py")
    print("═"*60 + "\n")


if __name__ == "__main__":
    generate_all()
