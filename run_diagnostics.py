#!/usr/bin/env python3
"""
CVA-SACS v6 — FULL DIAGNOSTIC PIPELINE
========================================
Runs every step of the pipeline and saves ALL intermediate
results to ./diagnostics/{TICKER}/ so you can inspect,
present, and explain every detail in a capstone review.

Usage:
    python run_diagnostics.py AAPL
    python run_diagnostics.py RELIANCE.NS
    python run_diagnostics.py AAPL MSFT NVDA    (multiple tickers)

Output (per ticker):
    diagnostics/AAPL/
    ├── 01_raw_data.csv                  ← Raw OHLCV from Yahoo Finance
    ├── 02_macro_data.csv                ← VIX, SPY, TLT, HYG, DXY, GLD
    ├── 03_features_full.csv             ← All 130 features (every row, every column)
    ├── 03_feature_list.txt              ← Names of all 130 features
    ├── 03_feature_stats.csv             ← Mean, std, min, max, NaN% per feature
    ├── 04_labels.csv                    ← risk_label + dir_label + components
    ├── 04_label_distribution.csv        ← Count per class (0-4)
    ├── 04_label_by_regime.csv           ← Label distribution per vol regime
    ├── 05_train_test_split.txt          ← Exact row indices for train/test
    ├── 06_model_config.json             ← Hyperparameters used for each model
    ├── 07_ensemble_predictions.csv      ← Every prediction: row, true, pred, proba[5], conf
    ├── 08_walkforward_summary.json      ← Overall accuracy, p-values, CI
    ├── 08_walkforward_per_window.csv    ← Accuracy per window with dates
    ├── 08_confusion_matrix.csv          ← 5x5 confusion matrix
    ├── 08_classification_report.json    ← Precision/recall/F1 per class
    ├── 08_significance_tests.json       ← Binomial test p-values
    ├── 08_confidence_sweep.csv          ← Accuracy vs coverage at each threshold
    ├── 09_backtest_trades.csv           ← Every trade: date, signal, return, PnL
    ├── 09_backtest_equity.csv           ← Daily equity curve
    ├── 09_backtest_summary.json         ← Sharpe, Sortino, DD, alpha, etc
    ├── 09_regime_performance.json       ← Performance per market regime
    ├── 09_benchmark_comparison.json     ← CVA-SACS vs MA vs RSI vs Random
    ├── 10_cri_breakdown.csv             ← CRI score + all 5 components for latest day
    ├── 10_cri_components_explained.txt  ← English explanation of CRI calculation
    ├── 11_feature_importance.csv        ← SHAP/gain importance ranked
    ├── 12_sacs_stress_test.json         ← SACS inputs, outputs, classification
    ├── 13_sentiment_scores.json         ← FinBERT headline scores (if available)
    └── SUMMARY.txt                      ← One-page human-readable summary
"""

import sys, os, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

# Add current dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cva_sacs_v6_ml import (
    FeatureEngineerV6, build_label_v6, EnsembleV6,
    WalkForwardV6, BacktestEngine, download_data, _fetch_macro,
    MODELS_DIR, save_model,
)

try:
    from cva_sacs_v6_ml import BenchmarkStrategies, permutation_test
    HAS_BENCHMARKS = True
except ImportError:
    HAS_BENCHMARKS = False

try:
    from scipy.stats import binom_test
    HAS_SCIPY = True
except ImportError:
    try:
        from scipy.stats import binomtest
        def binom_test(x, n, p, alternative="greater"):
            return binomtest(x, n, p, alternative=alternative).pvalue
        HAS_SCIPY = True
    except ImportError:
        HAS_SCIPY = False


def save_json(data, path):
    """Save dict/list to JSON, handling numpy types."""
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.bool_,)): return bool(obj)
        if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list): return [convert(v) for v in obj]
        return obj
    with open(path, "w") as f:
        json.dump(convert(data), f, indent=2, default=str)


def run_diagnostics(ticker: str):
    """Run full diagnostic pipeline for a single ticker."""
    
    out_dir = Path(f"diagnostics/{ticker.replace('.', '_')}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    summary_lines = []
    def log(msg):
        print(f"  {msg}")
        summary_lines.append(msg)
    
    print(f"\n{'='*65}")
    print(f"  CVA-SACS v6 DIAGNOSTICS — {ticker}")
    print(f"  Output: {out_dir}/")
    print(f"{'='*65}")
    
    # ══════════════════════════════════════════════════════════
    # STEP 1: RAW DATA
    # ══════════════════════════════════════════════════════════
    log(f"\n[STEP 1] Downloading raw OHLCV data for {ticker}...")
    df_raw = download_data(ticker, years=6)
    if df_raw is None or len(df_raw) < 400:
        log(f"  FAILED: Not enough data ({len(df_raw) if df_raw is not None else 0} rows)")
        return
    
    df_raw.to_csv(out_dir / "01_raw_data.csv", index=False)
    log(f"  Rows: {len(df_raw)}")
    log(f"  Date range: {df_raw['Date'].min().date()} → {df_raw['Date'].max().date()}")
    log(f"  Current price: {df_raw['Close'].iloc[-1]:.2f}")
    log(f"  Saved → 01_raw_data.csv")
    
    # ══════════════════════════════════════════════════════════
    # STEP 2: MACRO DATA
    # ══════════════════════════════════════════════════════════
    log(f"\n[STEP 2] Fetching macro data (VIX, SPY, TLT, HYG, DXY, GLD)...")
    macro = _fetch_macro()
    
    macro_df = pd.DataFrame({name: series for name, series in macro.items()})
    macro_df.to_csv(out_dir / "02_macro_data.csv")
    log(f"  Macro series: {list(macro.keys())}")
    for name, series in macro.items():
        log(f"    {name}: {len(series)} rows, latest={series.iloc[-1]:.2f}" if len(series) > 0 else f"    {name}: EMPTY")
    log(f"  Saved → 02_macro_data.csv")
    
    # ══════════════════════════════════════════════════════════
    # STEP 3: FEATURE ENGINEERING
    # ══════════════════════════════════════════════════════════
    log(f"\n[STEP 3] Building 130 features (8 layers: A→H)...")
    fe = FeatureEngineerV6()
    df_fe = fe.build(df_raw, macro=macro)
    fcs = fe.get_feature_cols(df_fe)
    fcs = [c for c in fcs if c in df_fe.columns]
    
    # Save full feature matrix
    df_fe.to_csv(out_dir / "03_features_full.csv", index=False)
    
    # Save feature list
    with open(out_dir / "03_feature_list.txt", "w") as f:
        f.write(f"Total features: {len(fcs)}\n\n")
        for i, fc in enumerate(fcs):
            f.write(f"{i+1:3d}. {fc}\n")
    
    # Save feature statistics
    stats = df_fe[fcs].describe().T
    stats["nan_pct"] = df_fe[fcs].isna().mean() * 100
    stats.to_csv(out_dir / "03_feature_stats.csv")
    
    log(f"  Total features: {len(fcs)}")
    log(f"  Feature matrix shape: {df_fe[fcs].shape}")
    log(f"  NaN% range: {stats['nan_pct'].min():.1f}% — {stats['nan_pct'].max():.1f}%")
    log(f"  Saved → 03_features_full.csv, 03_feature_list.txt, 03_feature_stats.csv")
    
    # ══════════════════════════════════════════════════════════
    # STEP 4: LABEL CONSTRUCTION
    # ══════════════════════════════════════════════════════════
    log(f"\n[STEP 4] Building dual-horizon labels (h1=5d, h2=10d)...")
    df_fe = build_label_v6(df_fe, h1=5, h2=10)
    df_fe = df_fe.dropna(subset=fcs + ["risk_label"]).reset_index(drop=True)
    
    # Save labels
    label_cols = ["Date", "Close", "risk_label", "dir_label"]
    df_fe[label_cols].to_csv(out_dir / "04_labels.csv", index=False)
    
    # Label distribution
    dist = df_fe["risk_label"].value_counts().sort_index()
    dist_df = pd.DataFrame({
        "class": dist.index,
        "name": [["CALM","MILD","MODERATE","ELEVATED","CRISIS"][int(c)] for c in dist.index],
        "count": dist.values,
        "percentage": (dist.values / dist.sum() * 100).round(1),
    })
    dist_df.to_csv(out_dir / "04_label_distribution.csv", index=False)
    
    # Label by vol regime
    if "vol_regime" in df_fe.columns:
        regime_label = df_fe.groupby(["vol_regime", "risk_label"]).size().unstack(fill_value=0)
        regime_label.to_csv(out_dir / "04_label_by_regime.csv")
    
    log(f"  Rows after labelling: {len(df_fe)}")
    log(f"  Label distribution:")
    for _, row in dist_df.iterrows():
        bar = "█" * int(row["percentage"] / 2)
        log(f"    {int(row['class'])} ({row['name']:<10}): {int(row['count']):4d} ({row['percentage']:5.1f}%) {bar}")
    imbalance = dist.max() / (dist.min() + 1)
    log(f"  Imbalance ratio: {imbalance:.1f}x {'⚠ WARNING' if imbalance > 3 else '✓ OK'}")
    log(f"  Saved → 04_labels.csv, 04_label_distribution.csv")
    
    if len(df_fe) < 600:
        log(f"\n  STOPPING: Not enough rows for walk-forward ({len(df_fe)} < 600)")
        return
    
    # ══════════════════════════════════════════════════════════
    # STEP 5: TRAIN/TEST SPLIT INFO
    # ══════════════════════════════════════════════════════════
    log(f"\n[STEP 5] Walk-forward split configuration...")
    wf = WalkForwardV6(min_train_days=504, step_days=5, test_window_days=5)
    splits = wf.get_splits(df_fe)
    
    with open(out_dir / "05_train_test_split.txt", "w") as f:
        f.write(f"Walk-Forward Configuration\n")
        f.write(f"{'='*50}\n")
        f.write(f"Min training days: 504 (~2 years)\n")
        f.write(f"Step size: 5 days\n")
        f.write(f"Test window: 5 days\n")
        f.write(f"Total windows: {len(splits)}\n")
        f.write(f"Total rows: {len(df_fe)}\n\n")
        f.write(f"First window:\n")
        f.write(f"  Train: rows 0-503 ({df_fe['Date'].iloc[0].date()} → {df_fe['Date'].iloc[503].date()})\n")
        f.write(f"  Test:  rows 504-508\n\n")
        f.write(f"Last window:\n")
        tr_last, te_last = splits[-1]
        f.write(f"  Train: rows 0-{tr_last[-1]} ({df_fe['Date'].iloc[0].date()} → {df_fe['Date'].iloc[tr_last[-1]].date()})\n")
        f.write(f"  Test:  rows {te_last[0]}-{te_last[-1]}\n")
    
    log(f"  Total walk-forward windows: {len(splits)}")
    log(f"  Saved → 05_train_test_split.txt")
    
    # ══════════════════════════════════════════════════════════
    # STEP 6: MODEL CONFIGURATION
    # ══════════════════════════════════════════════════════════
    log(f"\n[STEP 6] Model configuration...")
    model_config = {
        "ensemble_type": "EnsembleV6 (XGBoost + LightGBM + CatBoost + LR meta-learner)",
        "architecture": {
            "base_models": {
                "XGBoost": {"n_estimators": 1000, "max_depth": 5, "learning_rate": 0.04,
                           "subsample": 0.8, "colsample_bytree": 0.7},
                "LightGBM": {"n_estimators": 1000, "max_depth": 5, "learning_rate": 0.04,
                            "subsample": 0.8, "colsample_bytree": 0.7},
                "CatBoost": {"iterations": 800, "depth": 6, "learning_rate": 0.04},
            },
            "meta_learner": "LogisticRegression(C=0.5, multinomial)",
            "scaler": "RobustScaler",
        },
        "training_split": "60% train / 20% OOF (meta) / 20% test",
        "n_classes": 5,
        "class_names": {0: "CALM", 1: "MILD", 2: "MODERATE", 3: "ELEVATED", 4: "CRISIS"},
        "n_features": len(fcs),
        "n_training_rows": len(df_fe),
        "class_weighting": "inverse frequency (balanced)",
    }
    save_json(model_config, out_dir / "06_model_config.json")
    log(f"  Saved → 06_model_config.json")
    
    # ══════════════════════════════════════════════════════════
    # STEP 7-8: WALK-FORWARD VALIDATION
    # ══════════════════════════════════════════════════════════
    log(f"\n[STEP 7-8] Running walk-forward validation ({len(splits)} windows)...")
    log(f"  This trains ~{len(splits)} separate models. Please wait...")
    
    wf_res = wf.run(df_fe, fcs)
    
    if "error" in wf_res:
        log(f"  FAILED: {wf_res['error']}")
        return
    
    # Save predictions
    pred_df = pd.DataFrame({
        "date": wf_res["dates"],
        "true_label": wf_res["true_labels"],
        "predicted_label": wf_res["predictions"],
        "confidence": [round(c, 4) for c in wf_res["confidences"]],
        "correct": [int(t == p) for t, p in zip(wf_res["true_labels"], wf_res["predictions"])],
    })
    # Add class probabilities
    probas = np.array(wf_res["probabilities"])
    for i in range(min(5, probas.shape[1])):
        pred_df[f"prob_class_{i}"] = probas[:, i].round(4)
    pred_df.to_csv(out_dir / "07_ensemble_predictions.csv", index=False)
    
    # Walk-forward summary
    wf_summary = {
        "overall_accuracy": wf_res["overall_accuracy"],
        "directional_accuracy": wf_res["directional_accuracy"],
        "high_conf_accuracy": wf_res["high_conf_accuracy"],
        "high_conf_coverage": wf_res["high_conf_coverage"],
        "accuracy_95ci": wf_res["accuracy_95ci"],
        "n_windows": wf_res["n_windows"],
        "n_test_samples": wf_res["n_test_samples"],
    }
    save_json(wf_summary, out_dir / "08_walkforward_summary.json")
    
    # Per-window results
    if wf_res.get("window_results"):
        pd.DataFrame(wf_res["window_results"]).to_csv(
            out_dir / "08_walkforward_per_window.csv", index=False)
    
    # Confusion matrix
    if wf_res.get("confusion_matrix"):
        cm = np.array(wf_res["confusion_matrix"])
        labels = ["CALM", "MILD", "MOD", "ELEV", "CRISIS"][:cm.shape[0]]
        cm_df = pd.DataFrame(cm, index=[f"True_{l}" for l in labels],
                            columns=[f"Pred_{l}" for l in labels])
        cm_df.to_csv(out_dir / "08_confusion_matrix.csv")
    
    # Classification report
    if wf_res.get("classification_report"):
        save_json(wf_res["classification_report"], out_dir / "08_classification_report.json")
    
    # Significance tests
    if wf_res.get("significance_tests"):
        save_json(wf_res["significance_tests"], out_dir / "08_significance_tests.json")
    
    # Confidence sweep
    if wf_res.get("confidence_sweep"):
        pd.DataFrame(wf_res["confidence_sweep"]).to_csv(
            out_dir / "08_confidence_sweep.csv", index=False)
    
    log(f"  Overall accuracy:     {wf_res['overall_accuracy']:.3f}")
    log(f"  Directional accuracy: {wf_res['directional_accuracy']:.3f}")
    if wf_res.get("significance_tests", {}).get("overall"):
        sig = wf_res["significance_tests"]["overall"]
        log(f"  p-value vs random:    {sig['p_value']:.6f} {'✓ SIGNIFICANT' if sig['significant_at_05'] else '✗ NOT SIGNIFICANT'}")
    log(f"  95% CI:               {wf_res['accuracy_95ci']}")
    log(f"  Windows:              {wf_res['n_windows']}")
    log(f"  Test samples:         {wf_res['n_test_samples']}")
    log(f"  Saved → 07_ensemble_predictions.csv, 08_*.csv/json")
    
    # ══════════════════════════════════════════════════════════
    # STEP 9: BACKTEST
    # ══════════════════════════════════════════════════════════
    log(f"\n[STEP 9] Running backtest (long-only, Kelly sizing)...")
    bt = BacktestEngine(mode="long_only", sizing="kelly", friction=0.001)
    bt_res = bt.run(df_raw, wf_res)
    
    if "error" not in bt_res:
        # Trade log
        pd.DataFrame(bt_res["trades"]).to_csv(out_dir / "09_backtest_trades.csv", index=False)
        
        # Equity curve
        eq_df = pd.DataFrame({
            "date": bt_res["equity_dates"],
            "equity": bt_res["equity_curve"],
        })
        eq_df.to_csv(out_dir / "09_backtest_equity.csv", index=False)
        
        # Summary
        bt_summary = {k: v for k, v in bt_res.items() 
                     if k not in ("trades", "equity_curve", "equity_dates", "signal_breakdown")}
        bt_summary["signal_breakdown"] = bt_res.get("signal_breakdown", {})
        save_json(bt_summary, out_dir / "09_backtest_summary.json")
        
        # Regime performance
        if bt_res.get("regime_analysis"):
            save_json(bt_res["regime_analysis"], out_dir / "09_regime_performance.json")
        
        # Benchmark comparison
        if HAS_BENCHMARKS:
            try:
                benchmarks = BenchmarkStrategies.run_all(
                    df_raw.set_index("Date")["Close"],
                    cva_trades=bt_res["n_trades"]
                )
                benchmarks.append({
                    "strategy": "CVA-SACS v6",
                    "total_return": bt_res["total_return"],
                    "sharpe": bt_res["sharpe_ratio"],
                    "max_drawdown": bt_res["max_drawdown"],
                    "n_trades": bt_res["n_trades"],
                })
                save_json(benchmarks, out_dir / "09_benchmark_comparison.json")
                log(f"  Benchmark comparison:")
                for b in benchmarks:
                    log(f"    {b['strategy']:<20} return={b['total_return']:+.1%}  sharpe={b['sharpe']:+.2f}")
            except Exception as e:
                log(f"  Benchmark error: {e}")
        
        log(f"  Total return:    {bt_res['total_return']:+.1%}")
        log(f"  Annualised:      {bt_res['annualised_return']:+.1%}")
        log(f"  Buy & hold:      {bt_res['buy_hold_return']:+.1%}")
        log(f"  Alpha:           {bt_res['alpha']:+.1%}")
        log(f"  Sharpe:          {bt_res['sharpe_ratio']:.2f}")
        log(f"  Max drawdown:    {bt_res['max_drawdown']:.1%}")
        log(f"  Win rate:        {bt_res['win_rate']:.0%}")
        log(f"  Trades:          {bt_res['n_trades']}")
        log(f"  Saved → 09_backtest_trades.csv, 09_backtest_equity.csv, 09_backtest_summary.json")
    else:
        log(f"  Backtest failed: {bt_res['error']}")
    
    # ══════════════════════════════════════════════════════════
    # STEP 10: CRI BREAKDOWN
    # ══════════════════════════════════════════════════════════
    log(f"\n[STEP 10] CRI (Composite Risk Index) calculation for latest day...")
    
    c = df_raw["Close"]
    lr = np.log(c / c.shift(1))
    rsi_val = float((100 - 100 / (c.diff().clip(lower=0).rolling(14).mean() / 
              ((-c.diff().clip(upper=0)).rolling(14).mean() + 1e-9) + 1)).iloc[-1])
    vol_ann = float(lr.rolling(20).std().iloc[-1] * np.sqrt(252) * 100)
    var5 = float(abs(np.percentile(lr.dropna().tail(252), 5)) * 100)
    ret5d = float((c.iloc[-1] / c.iloc[-6] - 1) * 100) if len(c) > 5 else 0
    
    # SACS (fixed version)
    vol_1x = float(lr.rolling(20).std().iloc[-1])
    stressed_2x = lr * 2.0
    stressed_3x = lr * 3.0
    vol_2x = float(stressed_2x.rolling(20).std().iloc[-1])
    vol_3x = float(stressed_3x.rolling(20).std().iloc[-1])
    d2 = abs(vol_2x - vol_1x) / (vol_1x + 1e-9)
    d3 = abs(vol_3x - vol_1x) / (vol_1x + 1e-9)
    sacs_cls = "BREAKS" if d3 > 0.40 else "FRAGILE" if d2 > 0.20 else "ROBUST"
    
    # ML risk class (use last walk-forward prediction)
    ml_rs = int(wf_res["predictions"][-1]) if wf_res["predictions"] else 2
    
    # CRI components
    ml_c = ml_rs * 25.0
    var_c = min(100, var5 / 30 * 100)
    sacs_c = {"ROBUST": 10, "FRAGILE": 55, "BREAKS": 90}.get(sacs_cls, 50)
    mom_c = max(0, min(100, -ret5d * 10 + 50))
    vol_c = min(100, max(0, (vol_ann - 10) / 70 * 100))
    cri = round(0.30*ml_c + 0.25*var_c + 0.20*sacs_c + 0.15*mom_c + 0.10*vol_c, 1)
    cri = max(0, min(100, cri))
    zone = "SAFE" if cri < 26 else "CAUTION" if cri < 51 else "ELEVATED" if cri < 76 else "DANGER"
    
    cri_data = {
        "ticker": ticker,
        "date": str(df_raw["Date"].iloc[-1].date()),
        "price": float(c.iloc[-1]),
        "cri_score": cri,
        "cri_zone": zone,
        "components": {
            "ML_risk_class": {"value": ml_rs, "score": round(ml_c, 1), "weight": 0.30,
                             "contribution": round(0.30*ml_c, 1),
                             "explanation": f"ML predicted class {ml_rs} → score {ml_c:.0f} (0=safe, 4=crisis)"},
            "VaR_5pct": {"value": round(var5, 2), "score": round(var_c, 1), "weight": 0.25,
                        "contribution": round(0.25*var_c, 1),
                        "explanation": f"5th percentile daily loss = {var5:.2f}% → score {var_c:.0f}"},
            "SACS_stress": {"value": sacs_cls, "score": round(sacs_c, 1), "weight": 0.20,
                           "contribution": round(0.20*sacs_c, 1),
                           "explanation": f"Stress test = {sacs_cls} (d2={d2:.3f}, d3={d3:.3f}) → score {sacs_c:.0f}"},
            "Momentum_5d": {"value": round(ret5d, 2), "score": round(mom_c, 1), "weight": 0.15,
                           "contribution": round(0.15*mom_c, 1),
                           "explanation": f"5-day return = {ret5d:+.2f}% → score {mom_c:.0f} (negative return = higher risk)"},
            "Volatility_ann": {"value": round(vol_ann, 2), "score": round(vol_c, 1), "weight": 0.10,
                              "contribution": round(0.10*vol_c, 1),
                              "explanation": f"Annualised vol = {vol_ann:.1f}% → score {vol_c:.0f}"},
        },
        "formula": "CRI = 0.30×ML + 0.25×VaR + 0.20×SACS + 0.15×Mom + 0.10×Vol",
        "calculation": f"CRI = 0.30×{ml_c:.0f} + 0.25×{var_c:.0f} + 0.20×{sacs_c:.0f} + 0.15×{mom_c:.0f} + 0.10×{vol_c:.0f} = {cri}",
    }
    save_json(cri_data, out_dir / "10_cri_breakdown.json")
    
    # Human-readable CRI explanation
    with open(out_dir / "10_cri_components_explained.txt", "w") as f:
        f.write(f"CRI BREAKDOWN FOR {ticker} — {cri_data['date']}\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Price: ${cri_data['price']:,.2f}\n")
        f.write(f"CRI Score: {cri} / 100 → {zone}\n\n")
        f.write(f"Formula: CRI = 0.30×ML + 0.25×VaR + 0.20×SACS + 0.15×Mom + 0.10×Vol\n\n")
        f.write(f"Components:\n")
        f.write(f"{'-'*60}\n")
        for name, comp in cri_data["components"].items():
            f.write(f"\n  {name}:\n")
            f.write(f"    Raw value:    {comp['value']}\n")
            f.write(f"    Scaled score: {comp['score']:.1f} / 100\n")
            f.write(f"    Weight:       {comp['weight']}\n")
            f.write(f"    Contribution: {comp['contribution']:.1f} to CRI\n")
            f.write(f"    Explanation:  {comp['explanation']}\n")
        f.write(f"\n{'='*60}\n")
        f.write(f"Total: {cri_data['calculation']}\n")
        f.write(f"Zone thresholds: SAFE <26 | CAUTION 26-50 | ELEVATED 51-75 | DANGER >75\n")
    
    log(f"  CRI = {cri} → {zone}")
    for name, comp in cri_data["components"].items():
        log(f"    {name:<20} {comp['contribution']:.1f}  ({comp['explanation'][:50]})")
    log(f"  Saved → 10_cri_breakdown.json, 10_cri_components_explained.txt")
    
    # ══════════════════════════════════════════════════════════
    # STEP 11: FEATURE IMPORTANCE
    # ══════════════════════════════════════════════════════════
    log(f"\n[STEP 11] Computing feature importance...")
    try:
        ens = EnsembleV6()
        cut = int(len(df_fe) * 0.75)
        ens.fit(df_fe.iloc[:cut][fcs], df_fe.iloc[:cut]["risk_label"].values)
        fi = ens.feature_importance()
        if not fi.empty:
            fi.to_csv(out_dir / "11_feature_importance.csv", index=False)
            log(f"  Top 10 features:")
            for _, row in fi.head(10).iterrows():
                bar = "█" * int(row["combined"] * 300)
                log(f"    {row['feature']:<30} {bar} {row['combined']:.4f}")
            log(f"  Saved → 11_feature_importance.csv")
        
        # Save model for future use
        save_model(ticker, ens, fcs,
                  {"n_rows": len(df_fe), "n_features": len(fcs),
                   "wf_accuracy": wf_res.get("overall_accuracy")})
        log(f"  Model saved → models_v6/model_{ticker.replace('.','_')}.pkl")
    except Exception as e:
        log(f"  Feature importance error: {e}")
    
    # ══════════════════════════════════════════════════════════
    # STEP 12: SACS STRESS TEST DETAIL
    # ══════════════════════════════════════════════════════════
    log(f"\n[STEP 12] SACS stress test detail...")
    sacs_detail = {
        "ticker": ticker,
        "vol_1x_raw": round(float(vol_1x), 6),
        "vol_2x_stressed": round(float(vol_2x), 6),
        "vol_3x_stressed": round(float(vol_3x), 6),
        "d2_deviation": round(d2, 4),
        "d3_deviation": round(d3, 4),
        "classification": sacs_cls,
        "thresholds": {"FRAGILE": ">0.20 on d2", "BREAKS": ">0.40 on d3"},
        "method": "Multiply daily log returns by 2x/3x, recompute rolling 20d vol",
        "interpretation": f"Under 2x stress, vol deviates {d2:.1%} from baseline. Under 3x stress, {d3:.1%}. Classification: {sacs_cls}.",
    }
    save_json(sacs_detail, out_dir / "12_sacs_stress_test.json")
    log(f"  SACS: {sacs_cls} (d2={d2:.3f}, d3={d3:.3f})")
    log(f"  Saved → 12_sacs_stress_test.json")
    
    # ══════════════════════════════════════════════════════════
    # SUMMARY FILE
    # ══════════════════════════════════════════════════════════
    with open(out_dir / "SUMMARY.txt", "w") as f:
        f.write(f"CVA-SACS v6 DIAGNOSTIC SUMMARY — {ticker}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*65}\n\n")
        for line in summary_lines:
            f.write(line + "\n")
    
    print(f"\n{'='*65}")
    print(f"  DONE — All files saved to {out_dir}/")
    print(f"  Open SUMMARY.txt for the full narrative")
    print(f"  Open 10_cri_components_explained.txt for CRI walkthrough")
    print(f"{'='*65}")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_diagnostics.py AAPL [MSFT] [NVDA] ...")
        print("       python run_diagnostics.py RELIANCE.NS")
        sys.exit(1)
    
    tickers = [t.upper() for t in sys.argv[1:]]
    for ticker in tickers:
        run_diagnostics(ticker)
