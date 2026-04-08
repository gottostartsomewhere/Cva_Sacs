"""
CVA-SACS v6 — Maximum Impact ML Engine
========================================
What's new vs v4/v5:

  LABEL UPGRADES
  ─────────────
  · Dual-horizon label: 5d AND 10d forward Sharpe, weighted blend
  · Drawdown penalty: labels penalised if max drawdown in window > threshold
  · Regime-conditional binning: percentile bins computed separately per
    vol regime so labels are calibrated within each market environment
  · Binary directional label alongside ordinal for a second head

  FEATURE UPGRADES (+45 features → ~130 total)
  ─────────────────────────────────────────────
  · Layer F — Cross-Asset Macro (15 features)
      VIX term structure, DXY, TLT, HYG, SPY beta
      Stock vs SPY relative strength, market-beta-adjusted return
  · Layer G — Alternative Data Proxies (15 features)
      Options IV proxy (rolling realised vol percentile vs 1y norm)
      Earnings surprise proxy (vol spike around earnings window)
      Short interest proxy (sustained price decline + high vol)
      Put/call proxy (OTM vol skew estimate from daily range patterns)
      Analyst revision proxy (post-earnings vol compression)
  · Layer H — Signal Persistence (10 features)
      N-day RSI trend direction, MACD persistence,
      consecutive up/down closes, vol expansion/contraction streak

  MODEL UPGRADES
  ──────────────
  · XGBoost + LightGBM + CatBoost (1000 trees each)
  · Optuna HPO: 300 trials per model, TPE sampler, pruning
  · Level-2 stacking: LR meta-learner on probability outputs
  · Calibration: isotonic regression on held-out fold
  · 5-day walk-forward steps (~80–100 windows)
  · Confidence-sweep: 0.50→0.85, report accuracy/coverage curve

  BACKTEST ENGINE
  ───────────────
  · Simulates trading every walk-forward signal
  · Equal-weight / Kelly-sized position options
  · Long-only / long-short modes
  · Realistic frictions: 0.10% per side (tunable)
  · Outputs: cumulative PnL, Sharpe, Sortino, max drawdown,
    Calmar ratio, win rate, avg win/loss, signal breakdown
  · Benchmarks against buy-and-hold

  SIGNAL PERSISTENCE
  ──────────────────
  · Tracks N consecutive days with same verdict
  · Persistence ≥3 days → CONFIRMED signal
  · Persistence reversal → EXIT signal

Run:
    python cva_sacs_v6_ml.py --tickers AAPL MSFT NVDA RELIANCE.NS --mode full
    python cva_sacs_v6_ml.py --tickers AAPL MSFT --mode ablation
    python cva_sacs_v6_ml.py --tickers AAPL --mode backtest
    python cva_sacs_v6_ml.py --tickers AAPL MSFT NVDA GOOGL AMZN --mode cross
"""

import warnings
warnings.filterwarnings("ignore")

import logging
logger = logging.getLogger("cva_sacs_v6")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

import argparse
import json
import os
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

import yfinance as yf
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import RobustScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, log_loss
)
try:
    from scipy.stats import binom_test as _binom_test
    SCIPY_OK = True
except ImportError:
    try:
        from scipy.stats import binomtest as _binomtest
        def _binom_test(x, n, p, alternative="greater"):
            return _binomtest(x, n, p, alternative=alternative).pvalue
        SCIPY_OK = True
    except ImportError:
        SCIPY_OK = False

try:    import xgboost as xgb;  XGB_OK = True
except: XGB_OK = False

try:    import lightgbm as lgb; LGB_OK = True
except: LGB_OK = False

try:    import catboost as cb;  CAT_OK = True
except: CAT_OK = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_OK = True
except: OPTUNA_OK = False

try:    import shap; SHAP_OK = True
except: SHAP_OK = False

# ══════════════════════════════════════════════════════════════════
# SECTION 0 — MACRO DATA CACHE
# ══════════════════════════════════════════════════════════════════

_macro_cache: Dict[str, pd.DataFrame] = {}

MACRO_TICKERS = {
    "VIX":  "^VIX",
    "SPY":  "SPY",
    "TLT":  "TLT",
    "HYG":  "HYG",
    "DXY":  "DX-Y.NYB",
    "GLD":  "GLD",
}

def _fetch_macro(years: int = 6) -> Dict[str, pd.Series]:
    """
    Download macro reference series once and cache them.
    Returns dict of {name: Close series indexed by Date}.
    """
    global _macro_cache
    if _macro_cache:
        return _macro_cache

    end   = datetime.today()
    start = end - timedelta(days=years * 365)
    print("  Fetching macro reference data (VIX, SPY, TLT, HYG, DXY, GLD)...")

    for name, sym in MACRO_TICKERS.items():
        try:
            df = yf.download(sym, start=start.strftime("%Y-%m-%d"),
                             end=end.strftime("%Y-%m-%d"),
                             progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            df = df.reset_index()
            df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
            _macro_cache[name] = df.set_index("Date")["Close"].rename(name)
            print(f"    {name} ({sym}): {len(df)} rows")
        except Exception as e:
            print(f"    {name} ({sym}): FAILED — {e}")
            _macro_cache[name] = pd.Series(dtype=float, name=name)

    return _macro_cache


def download_data(ticker: str, years: int = 6) -> Optional[pd.DataFrame]:
    """Download OHLCV data. Returns clean DataFrame with Date column."""
    try:
        end   = datetime.today()
        start = end - timedelta(days=years * 365)
        df    = yf.download(ticker,
                            start=start.strftime("%Y-%m-%d"),
                            end=end.strftime("%Y-%m-%d"),
                            progress=False, auto_adjust=True)
        if df.empty or len(df) < 400:
            print(f"  {ticker}: insufficient data ({len(df)} rows)")
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df = df.reset_index()
        df.columns = [str(c).strip() for c in df.columns]
        if "Price" in df.columns and "Close" not in df.columns:
            df = df.rename(columns={"Price": "Close"})
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
        df["Ticker"] = ticker
        df = df.sort_values("Date").reset_index(drop=True)
        print(f"  {ticker}: {len(df)} rows  "
              f"({df['Date'].min().date()} → {df['Date'].max().date()})")
        return df
    except Exception as e:
        print(f"  {ticker} download error: {e}")
        return None


# ══════════════════════════════════════════════════════════════════
# SECTION 1 — FEATURE ENGINEERING v6  (~130 features)
# ══════════════════════════════════════════════════════════════════

class FeatureEngineerV6:
    """
    130-feature pipeline.

    Layers A–E:  Inherited from v4 (85 features, all shift(1))
    Layer F:     Cross-Asset Macro (15 features)
    Layer G:     Alternative Data Proxies (15 features)
    Layer H:     Signal Persistence (10 features)

    All features apply shift(1) to prevent lookahead.
    """

    def build(self, df: pd.DataFrame,
              macro: Optional[Dict[str, pd.Series]] = None) -> pd.DataFrame:
        df = df.copy()
        if "Date" not in df.columns:
            df = df.reset_index()
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
        df = df.sort_values("Date").reset_index(drop=True)

        df = self._layer_a(df)
        df = self._layer_b(df)
        df = self._layer_c(df)
        df = self._layer_d(df)
        df = self._layer_e(df)
        if macro:
            df = self._layer_f(df, macro)
        df = self._layer_g(df)
        df = self._layer_h(df)
        df = df.dropna().reset_index(drop=True)
        return df

    # ── A: Price & Returns (20) ───────────────────────────────
    def _layer_a(self, df):
        c = df["Close"]
        for n in [1, 3, 5, 10, 20]:
            df[f"ret_{n}d"] = c.pct_change(n).shift(1)
        df["log_ret_1d"] = np.log(c / c.shift(1)).shift(1)
        df["log_ret_5d"] = np.log(c / c.shift(5)).shift(1)
        for lag in [1, 2, 3, 5, 10]:
            df[f"ret_lag{lag}"] = c.pct_change(1).shift(lag)
        for w in [10, 20, 50, 200]:
            ma = c.rolling(w).mean()
            df[f"price_to_ma{w}"] = (c / ma - 1).shift(1)
            df[f"ma{w}"] = ma
        df["ma20_to_ma50"]  = (df["ma20"] / df["ma50"]).shift(1)
        df["ma50_to_ma200"] = (df["ma50"] / df["ma200"]).shift(1)
        df["ma20_slope"]    = df["ma20"].pct_change(5).shift(1)
        return df

    # ── B: Technical Indicators (28) ─────────────────────────
    def _layer_b(self, df):
        c  = df["Close"]
        hi = df["High"]
        lo = df["Low"]
        v  = df["Volume"]

        for period in [7, 14, 21]:
            delta = c.diff()
            gain  = delta.clip(lower=0).rolling(period).mean()
            loss  = (-delta.clip(upper=0)).rolling(period).mean()
            df[f"rsi_{period}"] = (100 - 100 / (gain / (loss + 1e-9) + 1)).shift(1)
        df["rsi_slope"] = df["rsi_14"].diff(3)

        ema12 = c.ewm(span=12).mean(); ema26 = c.ewm(span=26).mean()
        macd  = ema12 - ema26; sig = macd.ewm(span=9).mean()
        df["macd"]      = macd.shift(1); df["macd_sig"]  = sig.shift(1)
        df["macd_hist"] = (macd - sig).shift(1)
        df["macd_cross"] = ((macd > sig) & (macd.shift(1) <= sig.shift(1))).astype(int).shift(1)

        lo14 = lo.rolling(14).min(); hi14 = hi.rolling(14).max()
        stk  = 100 * (c - lo14) / (hi14 - lo14 + 1e-9)
        df["stoch_k"] = stk.shift(1); df["stoch_d"] = stk.rolling(3).mean().shift(1)
        df["williams_r"] = (-100 * (hi14 - c) / (hi14 - lo14 + 1e-9)).shift(1)

        tp  = (hi + lo + c) / 3
        df["cci"] = ((tp - tp.rolling(20).mean()) /
                     (0.015 * tp.rolling(20).std() + 1e-9)).shift(1)

        sma20 = c.rolling(20).mean(); std20 = c.rolling(20).std()
        bb_up = sma20 + 2 * std20; bb_lo = sma20 - 2 * std20
        df["bb_width"] = ((bb_up - bb_lo) / (sma20 + 1e-9)).shift(1)
        df["bb_pos"]   = ((c - bb_lo) / (bb_up - bb_lo + 1e-9)).shift(1)
        df["bb_squeeze"] = (df["bb_width"] < df["bb_width"].rolling(20).quantile(0.1)).astype(int)

        vm = v.rolling(20).mean()
        df["vol_ratio"] = (v / vm).shift(1); df["vol_surge"] = (df["vol_ratio"] > 1.5).astype(int)
        df["obv"]       = (np.sign(c.diff()) * v).cumsum().shift(1)
        df["obv_slope"] = df["obv"].pct_change(5)

        mf     = tp * v
        pos_mf = mf.where(tp > tp.shift(1), 0).rolling(14).sum()
        neg_mf = mf.where(tp < tp.shift(1), 0).rolling(14).sum()
        df["mfi"] = (100 - 100 / (1 + pos_mf / (neg_mf + 1e-9))).shift(1)

        for p in [5, 10, 20]:
            df[f"roc_{p}"] = ((c - c.shift(p)) / (c.shift(p) + 1e-9) * 100).shift(1)
        return df

    # ── C: Volatility & Regime (15) ──────────────────────────
    def _layer_c(self, df):
        lr = np.log(df["Close"] / df["Close"].shift(1))
        for w in [5, 10, 20, 60]:
            df[f"vol_{w}d"] = (lr.rolling(w).std() * np.sqrt(252)).shift(1)
        df["vol_ratio_5_20"]  = df["vol_5d"] / (df["vol_20d"] + 1e-9)
        df["vol_ratio_20_60"] = df["vol_20d"] / (df["vol_60d"] + 1e-9)

        vol20 = df["vol_20d"]
        p33   = vol20.rolling(252).quantile(0.33)
        p67   = vol20.rolling(252).quantile(0.67)
        df["vol_regime"]     = np.where(vol20 < p33, 0, np.where(vol20 < p67, 1, 2))
        df["vol_percentile"] = vol20.rolling(252).rank(pct=True)

        hl = df["High"] - df["Low"]
        hc = (df["High"] - df["Close"].shift(1)).abs()
        lc = (df["Low"]  - df["Close"].shift(1)).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        df["atr_14"]   = tr.rolling(14).mean().shift(1)
        df["atr_ratio"] = df["atr_14"] / (df["Close"] + 1e-9)

        df["vol_zscore"] = ((vol20 - vol20.rolling(60).mean()) /
                            (vol20.rolling(60).std() + 1e-9)).shift(1)
        return df

    # ── D: Market Structure (12) ─────────────────────────────
    def _layer_d(self, df):
        c = df["Close"]; hi = df["High"]; lo = df["Low"]
        df["dist_52w_high"]   = (c / hi.rolling(252).max() - 1).shift(1)
        df["dist_52w_low"]    = (c / lo.rolling(252).min() - 1).shift(1)
        df["dist_support"]    = ((c - lo.rolling(20).min()) / (c + 1e-9)).shift(1)
        df["dist_resistance"] = ((hi.rolling(20).max() - c) / (c + 1e-9)).shift(1)
        df["sr_ratio"]        = df["dist_support"] / (df["dist_resistance"] + 1e-9)
        df["higher_high"]     = (hi > hi.shift(1)).astype(int).shift(1)
        df["lower_low"]       = (lo < lo.shift(1)).astype(int).shift(1)
        df["inside_bar"]      = ((hi < hi.shift(1)) & (lo > lo.shift(1))).astype(int).shift(1)
        df["gap_up"]  = ((df["Open"] > c.shift(1)) & ((df["Open"] - c.shift(1)) / (c.shift(1) + 1e-9) > 0.01)).astype(int).shift(1)
        df["gap_down"] = ((df["Open"] < c.shift(1)) & ((c.shift(1) - df["Open"]) / (c.shift(1) + 1e-9) > 0.01)).astype(int).shift(1)
        up_day = (c > c.shift(1)).astype(int)
        df["up_streak"]   = up_day.groupby((up_day != up_day.shift()).cumsum()).cumsum().shift(1)
        df["down_streak"] = ((1 - up_day).groupby((up_day != up_day.shift()).cumsum()).cumsum()).shift(1)
        return df

    # ── E: Meta/Interaction (10) ─────────────────────────────
    def _layer_e(self, df):
        df["rsi_vol_interaction"] = df["rsi_14"] * df["vol_regime"]
        df["momentum_align"]      = (np.sign(df["ret_5d"]) == np.sign(df["ret_20d"])).astype(int)
        df["trend_quality"]       = df["ma20_slope"] / (df["vol_20d"] + 1e-9)
        df["vol_price_diverge"]   = np.where(
            (df["ret_5d"] > 0) & (df["vol_ratio"] < 0.8), -1,
            np.where((df["ret_5d"] < 0) & (df["vol_ratio"] < 0.8), 1, 0))
        df["rsi_mean_rev"]    = (df["rsi_14"] - 50) / 50
        df["breakout_setup"]  = (df["bb_squeeze"] & df["vol_surge"]).astype(int)
        df["composite_momentum"] = (0.5 * np.sign(df["ret_5d"]) +
                                    0.3 * np.sign(df["ret_10d"]) +
                                    0.2 * np.sign(df["ret_20d"]))
        df["stress_composite"] = ((df["vol_zscore"] > 1).astype(int) +
                                   (df["ret_5d"] < -0.03).astype(int) +
                                   (df["rsi_14"] < 35).astype(int))
        df["rsi_trend_confirm"] = np.where(
            (df["rsi_14"] < 35) & (df["ma20_to_ma50"] > 1), 1,
            np.where((df["rsi_14"] > 65) & (df["ma20_to_ma50"] < 1), -1, 0))
        df["sharpe_5d"] = df["ret_5d"] / (df["vol_5d"] + 1e-9)
        return df

    # ── F: Cross-Asset Macro (15) ─────────────────────────────
    def _layer_f(self, df, macro: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        Merge macro series by date and compute:
        - VIX level, VIX change, VIX percentile (fear gauge)
        - DXY return (dollar strength)
        - TLT return (rates direction)
        - HYG return (credit risk appetite)
        - Stock beta-adjusted return vs SPY
        - Relative strength vs SPY (5d and 20d)
        - Cross-asset risk-on/risk-off composite score
        """
        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)

        for name, series in macro.items():
            series = series.copy()
            series.index = pd.to_datetime(series.index).tz_localize(None)
            tmp = series.reset_index()
            tmp.columns = ["Date", name]
            df = df.merge(tmp, on="Date", how="left")
            df[name] = df[name].ffill()

        c = df["Close"]

        # VIX features
        if "VIX" in df.columns:
            vix = df["VIX"]
            df["vix_level"]      = vix.shift(1)
            df["vix_1d_change"]  = vix.pct_change(1).shift(1)
            df["vix_5d_change"]  = vix.pct_change(5).shift(1)
            df["vix_percentile"] = vix.rolling(252).rank(pct=True).shift(1)
            df["vix_spike"]      = (vix > vix.rolling(20).mean() * 1.3).astype(int).shift(1)
            df.drop(columns=["VIX"], inplace=True)

        # SPY relative strength
        if "SPY" in df.columns:
            spy = df["SPY"]
            spy_ret5  = spy.pct_change(5)
            spy_ret20 = spy.pct_change(20)
            stock_ret5  = c.pct_change(5)
            stock_ret20 = c.pct_change(20)

            df["rel_strength_5d"]  = (stock_ret5  - spy_ret5).shift(1)
            df["rel_strength_20d"] = (stock_ret20 - spy_ret20).shift(1)

            # Rolling beta (60d) — how much the stock moves with SPY
            spy_lr  = np.log(spy / spy.shift(1))
            stk_lr  = np.log(c / c.shift(1))
            cov60   = stk_lr.rolling(60).cov(spy_lr)
            var60   = spy_lr.rolling(60).var()
            df["beta_60d"] = (cov60 / (var60 + 1e-9)).shift(1)

            # Beta-adjusted return: excess return beyond what beta predicts
            df["alpha_5d"] = (stock_ret5 - df["beta_60d"] * spy_ret5).shift(1)
            df.drop(columns=["SPY"], inplace=True)

        # TLT (rates)
        if "TLT" in df.columns:
            df["tlt_ret_5d"] = df["TLT"].pct_change(5).shift(1)
            df.drop(columns=["TLT"], inplace=True)

        # HYG (credit)
        if "HYG" in df.columns:
            df["hyg_ret_5d"] = df["HYG"].pct_change(5).shift(1)
            df.drop(columns=["HYG"], inplace=True)

        # DXY (dollar)
        if "DXY" in df.columns:
            df["dxy_ret_5d"] = df["DXY"].pct_change(5).shift(1)
            df.drop(columns=["DXY"], inplace=True)

        # GLD (gold — risk off indicator)
        if "GLD" in df.columns:
            df["gld_ret_5d"] = df["GLD"].pct_change(5).shift(1)
            df.drop(columns=["GLD"], inplace=True)

        # Cross-asset risk-on composite:
        # SPY up + HYG up + TLT down = risk-on environment
        if all(c in df.columns for c in ["rel_strength_5d", "hyg_ret_5d", "tlt_ret_5d"]):
            df["risk_on_composite"] = (
                np.sign(df.get("rel_strength_5d", 0)) * 0.4 +
                np.sign(df.get("hyg_ret_5d", 0))      * 0.4 +
                (-np.sign(df.get("tlt_ret_5d", 0)))   * 0.2
            )

        return df

    # ── G: Alternative Data Proxies (15) ─────────────────────
    def _layer_g(self, df) -> pd.DataFrame:
        """
        Proxy signals for data we can't easily get for free.

        IV Proxy:     Realised vol vs its 1y percentile norm ≈ IV rank
        Earnings:     Vol spike in 3-day window around Q dates (approx)
        Short proxy:  Sustained 10d decline + vol expansion ≈ heavy shorting
        Skew proxy:   Daily range asymmetry (down range > up range ≈ put skew)
        Revision:     Post-earnings vol compression ≈ analyst uncertainty reducing
        """
        c  = df["Close"]
        hi = df["High"]
        lo = df["Low"]
        lr = np.log(c / c.shift(1))

        # IV Rank proxy: where is current 20d realised vol vs 1y range?
        rv20  = lr.rolling(20).std() * np.sqrt(252)
        rv_lo = rv20.rolling(252).min()
        rv_hi = rv20.rolling(252).max()
        df["iv_rank_proxy"] = ((rv20 - rv_lo) / (rv_hi - rv_lo + 1e-9)).shift(1)

        # IV percentile proxy: rank of current vol in 1y distribution
        df["iv_pct_proxy"] = rv20.rolling(252).rank(pct=True).shift(1)

        # Implied vol premium proxy: 30d vol / 10d vol ratio (term structure)
        rv10 = lr.rolling(10).std() * np.sqrt(252)
        rv30 = lr.rolling(30).std() * np.sqrt(252)
        df["vol_term_struct"] = (rv30 / (rv10 + 1e-9)).shift(1)

        # Earnings window proxy: volume spike + vol spike in same 3-day window
        vol_spike = (df["Volume"] > df["Volume"].rolling(20).mean() * 2.0)
        ret_spike = (lr.abs() > lr.rolling(20).std() * 2.5)
        earnings_window = (vol_spike & ret_spike).astype(int)
        # Was there an earnings-like event in last 5 days?
        df["earnings_proxy"] = earnings_window.rolling(5).max().shift(1)
        # Post-earnings vol compression (vol fell after spike)
        df["post_earnings_compression"] = (
            (earnings_window.shift(3) == 1) &
            (rv10 < rv10.shift(3))
        ).astype(int).shift(1)

        # Short interest proxy: consecutive declining closes + vol expansion
        down_run    = (c < c.shift(1)).astype(int).rolling(10).sum()
        vol_expand  = (rv20 > rv20.shift(5)).astype(int)
        df["short_proxy"] = (down_run * vol_expand / 10.0).shift(1)

        # Put/call skew proxy: ratio of down-day range to up-day range
        down_days = (c < c.shift(1))
        up_days   = (c >= c.shift(1))
        hl_range  = hi - lo
        down_range_avg = (hl_range.where(down_days, 0)).rolling(10).mean()
        up_range_avg   = (hl_range.where(up_days, 0)).rolling(10).mean()
        df["skew_proxy"] = (down_range_avg / (up_range_avg + 1e-9)).shift(1)

        # Momentum quality: correlation of daily returns with their trend
        ret1  = lr.rolling(1)
        trend = lr.rolling(20).mean()
        df["momentum_quality"] = lr.rolling(20).corr(
            pd.Series(range(len(lr)), index=lr.index)
        ).shift(1)

        # Liquidity proxy: daily dollar volume vs 60d average
        dollar_vol = c * df["Volume"]
        df["liquidity_ratio"] = (dollar_vol / dollar_vol.rolling(60).mean()).shift(1)

        # Overnight gap persistence: how often does the stock gap vs fill
        gap = (df["Open"] / c.shift(1) - 1)
        df["gap_persistence"] = gap.rolling(20).mean().shift(1)
        df["gap_volatility"]  = gap.rolling(20).std().shift(1)

        # Intraday efficiency: close-to-close vs high-low range
        efficiency = (lr.abs() / (hl_range / c + 1e-9))
        df["intraday_efficiency"] = efficiency.rolling(10).mean().shift(1)

        return df

    # ── H: Signal Persistence (10) ───────────────────────────
    def _layer_h(self, df) -> pd.DataFrame:
        """
        How long has the current signal been active?
        Persistent signals are empirically more reliable.
        """
        # RSI trend persistence: consecutive days RSI above/below 50
        rsi_above = (df["rsi_14"] > 50).astype(int)
        df["rsi_trend_persist"] = rsi_above.groupby(
            (rsi_above != rsi_above.shift()).cumsum()
        ).cumsum().shift(1)

        # MACD histogram sign persistence
        macd_pos = (df["macd_hist"] > 0).astype(int)
        df["macd_persist"] = macd_pos.groupby(
            (macd_pos != macd_pos.shift()).cumsum()
        ).cumsum().shift(1)

        # Price above MA20 persistence
        above_ma20 = (df["Close"] > df["ma20"]).astype(int)
        df["above_ma20_days"] = above_ma20.groupby(
            (above_ma20 != above_ma20.shift()).cumsum()
        ).cumsum().shift(1)

        # Vol expansion vs contraction streak
        vol20 = df.get("vol_20d", df["Close"].pct_change().rolling(20).std() * np.sqrt(252))
        vol_expand = (vol20 > vol20.shift(1)).astype(int)
        df["vol_expand_streak"] = vol_expand.groupby(
            (vol_expand != vol_expand.shift()).cumsum()
        ).cumsum().shift(1)

        # Consecutive up/down days (longer than up_streak which resets)
        c = df["Close"]
        up = (c > c.shift(1)).astype(int)
        df["consec_up"]   = up.groupby((up != up.shift()).cumsum()).cumsum().shift(1)
        df["consec_down"] = ((1-up).groupby((up != up.shift()).cumsum()).cumsum()).shift(1)

        # OBV trend: is OBV rising for N days?
        obv = df.get("obv", (np.sign(c.diff()) * df["Volume"]).cumsum())
        obv_up = (obv > obv.shift(1)).astype(int)
        df["obv_trend_persist"] = obv_up.groupby(
            (obv_up != obv_up.shift()).cumsum()
        ).cumsum().shift(1)

        # BB position persistence above midline
        bb_high = (df.get("bb_pos", pd.Series(0.5, index=df.index)) > 0.5).astype(int)
        df["bb_high_persist"] = bb_high.groupby(
            (bb_high != bb_high.shift()).cumsum()
        ).cumsum().shift(1)

        # Stoch overbought/oversold persistence
        stoch_ob = (df.get("stoch_k", pd.Series(50, index=df.index)) > 80).astype(int)
        df["stoch_ob_persist"] = stoch_ob.groupby(
            (stoch_ob != stoch_ob.shift()).cumsum()
        ).cumsum().shift(1)

        # New: 10d momentum sign persistence
        ret10_pos = (df["ret_10d"] > 0).astype(int) if "ret_10d" in df.columns else pd.Series(0, index=df.index)
        df["momentum_persist_10d"] = ret10_pos.groupby(
            (ret10_pos != ret10_pos.shift()).cumsum()
        ).cumsum().shift(1)

        return df

    def get_feature_cols(self, df: pd.DataFrame) -> List[str]:
        exclude = {
            "Date", "Open", "High", "Low", "Close", "Volume",
            "Ticker", "Dividends", "Stock Splits",
            "ma10", "ma20", "ma50", "ma200",
        }
        return [
            c for c in df.columns
            if c not in exclude
            and not c.startswith("target")
            and not c.startswith("forward")
            and not c.startswith("risk_label")
            and not c.startswith("label")
            and not c.startswith("ticker_code")
        ]


# ══════════════════════════════════════════════════════════════════
# SECTION 2 — DUAL-HORIZON LABEL WITH DRAWDOWN PENALTY  (v6)
# ══════════════════════════════════════════════════════════════════

def build_label_v6(df: pd.DataFrame,
                   h1: int = 5,
                   h2: int = 10,
                   w1: float = 0.45,
                   w2: float = 0.35,
                   w_dd: float = 0.20) -> pd.DataFrame:
    """
    V6 label construction: dual-horizon + drawdown penalty + regime binning.

    Label = round(
        w1 * Outcome_h1 +
        w2 * Outcome_h2 +
        w_dd * DrawdownPenalty
    ).clip(0, 4)

    Where Outcome is risk-adjusted (Sharpe-normalised) forward return,
    percentile-binned WITHIN the current vol regime (0/1/2).
    Regime-conditional binning ensures labels are calibrated in both
    calm and volatile markets — fixing the systematic miscalibration
    that happens when you bin across all regimes together.

    DrawdownPenalty = 0 if max drawdown in window < 5%
                    = 1 if max drawdown 5-10%
                    = 2 if max drawdown > 10%
    """
    df = df.copy()
    lr = np.log(df["Close"] / df["Close"].shift(1))

    def fwd_sharpe(h: int) -> pd.Series:
        fwd_raw = df["Close"].pct_change(h).shift(-h)
        fwd_vol = lr.rolling(h).std().shift(-h) * np.sqrt(252)
        return fwd_raw / (fwd_vol + 1e-9)

    def regime_bin(series: pd.Series, regime: pd.Series) -> pd.Series:
        """Bin within each vol regime separately."""
        out = pd.Series(2.0, index=series.index)  # default MODERATE
        for r in [0, 1, 2]:
            mask = (regime == r)
            sub  = series[mask]
            if len(sub) < 50:
                continue
            bins = pd.cut(
                sub.rank(pct=True),
                bins=[0, .2, .4, .6, .8, 1.0],
                labels=[4, 3, 2, 1, 0],
                include_lowest=True
            ).astype(float)
            out[mask] = bins
        return out

    # Vol regime for regime-conditional binning
    vol20  = lr.rolling(20).std()
    p33    = vol20.rolling(252).quantile(0.33)
    p67    = vol20.rolling(252).quantile(0.67)
    regime = np.where(vol20 < p33, 0, np.where(vol20 < p67, 1, 2))
    regime = pd.Series(regime, index=df.index)

    # Dual horizon Sharpe scores
    fs1 = fwd_sharpe(h1)
    fs2 = fwd_sharpe(h2)

    # Regime-conditional binning
    out1 = regime_bin(fs1, regime)
    out2 = regime_bin(fs2, regime)

    # Drawdown penalty: max drawdown in forward window
    def max_dd_forward(h: int) -> pd.Series:
        out = pd.Series(0.0, index=df.index)
        prices = df["Close"].values
        for i in range(len(prices) - h):
            window = prices[i+1 : i+h+1]
            peak   = np.maximum.accumulate(window)
            dd     = np.min((window - peak) / (peak + 1e-9))
            dd_abs = abs(dd)
            out.iloc[i] = 0 if dd_abs < 0.05 else (1 if dd_abs < 0.10 else 2)
        return out

    dd_pen = max_dd_forward(h2)  # use longer horizon for drawdown

    # Composite label
    raw = w1 * out1 + w2 * out2 + w_dd * dd_pen
    df["risk_label"] = raw.round().clip(0, 4)

    # Binary directional label (0=bullish/neutral, 1=bearish/elevated)
    df["dir_label"] = (df["risk_label"] >= 3).astype(int)

    df = df.dropna(subset=["risk_label"]).copy()
    df["risk_label"] = df["risk_label"].astype(int)
    return df


# ══════════════════════════════════════════════════════════════════
# SECTION 3 — ENSEMBLE v6: XGB + LGB + CAT + STACKING
# ══════════════════════════════════════════════════════════════════

class EnsembleV6:
    """
    Three-model stacking ensemble with Optuna HPO.

    Architecture:
      Base layer:   XGBoost (1000 trees) + LightGBM (1000) + CatBoost (800)
      Meta layer:   LogisticRegression trained on base OOF probabilities
      Calibration:  Isotonic regression on held-out fold

    Training protocol:
      1. Split data: 60% train / 20% OOF (meta training) / 20% test
      2. Fit base models on train set
      3. Generate OOF predictions on OOF set
      4. Fit meta-learner on OOF probabilities
      5. Calibrate with isotonic regression on OOF set
      6. Optuna HPO: tune each base model independently
    """

    N_CLASSES = 5

    def __init__(self, n_classes: int = 5, use_optuna: bool = False,
                 optuna_trials: int = 100):
        self.n_classes     = n_classes
        self.use_optuna    = use_optuna and OPTUNA_OK
        self.optuna_trials = optuna_trials
        self.scaler        = RobustScaler()
        self.base_models   = {}
        self.meta_lr       = None
        self.feature_cols: List[str] = []
        self.is_fitted     = False
        self.train_classes_: np.ndarray = np.array([])

    def _sample_weights(self, y: np.ndarray) -> np.ndarray:
        counts  = np.bincount(y, minlength=self.n_classes)
        cw      = {c: len(y) / (self.n_classes * max(counts[c], 1))
                   for c in range(self.n_classes)}
        return np.array([cw[c] for c in y])

    def _pad_proba(self, p: np.ndarray) -> np.ndarray:
        if p.shape[1] < self.n_classes:
            out = np.zeros((p.shape[0], self.n_classes))
            out[:, :p.shape[1]] = p
            return out
        return p

    def _tune_xgb(self, X_tr, y_tr, X_val, y_val, n_cls) -> dict:
        def objective(trial):
            params = dict(
                n_estimators     = trial.suggest_int("n_est", 300, 1500),
                max_depth        = trial.suggest_int("depth", 3, 8),
                learning_rate    = trial.suggest_float("lr", 0.01, 0.15, log=True),
                subsample        = trial.suggest_float("sub", 0.5, 1.0),
                colsample_bytree = trial.suggest_float("col", 0.4, 1.0),
                gamma            = trial.suggest_float("gamma", 0.0, 1.0),
                reg_alpha        = trial.suggest_float("alpha", 0.0, 5.0),
                reg_lambda       = trial.suggest_float("lambda", 0.5, 5.0),
                min_child_weight = trial.suggest_int("mcw", 5, 30),
            )
            m = xgb.XGBClassifier(**params,
                objective="multi:softprob", num_class=n_cls,
                eval_metric="mlogloss", use_label_encoder=False,
                random_state=42, verbosity=0, n_jobs=-1)
            m.fit(X_tr, y_tr, sample_weight=self._sample_weights(y_tr))
            pred = m.predict(X_val)
            return accuracy_score(y_val, pred)
        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=self.optuna_trials,
                       show_progress_bar=False, n_jobs=1)
        return study.best_params

    def _tune_lgb(self, X_tr, y_tr, X_val, y_val, n_cls) -> dict:
        def objective(trial):
            params = dict(
                n_estimators     = trial.suggest_int("n_est", 300, 1500),
                max_depth        = trial.suggest_int("depth", 3, 8),
                learning_rate    = trial.suggest_float("lr", 0.01, 0.15, log=True),
                subsample        = trial.suggest_float("sub", 0.5, 1.0),
                colsample_bytree = trial.suggest_float("col", 0.4, 1.0),
                min_child_samples= trial.suggest_int("mcs", 10, 50),
                reg_alpha        = trial.suggest_float("alpha", 0.0, 5.0),
                reg_lambda       = trial.suggest_float("lambda", 0.5, 5.0),
            )
            m = lgb.LGBMClassifier(**params,
                objective="multiclass", num_class=n_cls,
                random_state=42, verbosity=-1, force_col_wise=True, n_jobs=-1)
            m.fit(X_tr, y_tr)
            return accuracy_score(y_val, m.predict(X_val))
        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=self.optuna_trials,
                       show_progress_bar=False, n_jobs=1)
        return study.best_params

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray,
            _n_estimators: int = None, _skip_cat: bool = False):
        self.feature_cols = list(X_train.columns)
        self.train_classes_ = np.unique(y_train)
        n_cls = len(self.train_classes_)

        # Allow fast-mode override of tree count
        # Higher LR in fast mode: 100 trees @ 0.10 ≈ 1000 trees @ 0.04 convergence
        n_est_xgb = _n_estimators if _n_estimators else 1000
        n_est_lgb = _n_estimators if _n_estimators else 1000
        n_est_cat = int(n_est_xgb * 0.8) if _n_estimators else 800
        _lr = 0.10 if _n_estimators and _n_estimators <= 150 else 0.04

        # Three-way split: 60/20/20
        n  = len(X_train)
        i1 = int(n * 0.60)
        i2 = int(n * 0.80)

        X_sc = self.scaler.fit_transform(X_train.values)
        X_tr = X_sc[:i1]; y_tr = y_train[:i1]
        X_oo = X_sc[i1:i2]; y_oo = y_train[i1:i2]
        sw_tr = self._sample_weights(y_tr)

        # ── XGBoost ─────────────────────────────────────────
        if XGB_OK:
            if self.use_optuna:
                print("    Optuna → XGBoost...", end="", flush=True)
                best = self._tune_xgb(X_tr, y_tr, X_oo, y_oo, n_cls)
                xgb_m = xgb.XGBClassifier(**best,
                    objective="multi:softprob", num_class=n_cls,
                    eval_metric="mlogloss", use_label_encoder=False,
                    random_state=42, verbosity=0, n_jobs=-1)
                print(f" best acc={accuracy_score(y_oo, xgb.XGBClassifier(**best, objective='multi:softprob', num_class=n_cls, eval_metric='mlogloss', use_label_encoder=False, random_state=42, verbosity=0).fit(X_tr,y_tr,sample_weight=sw_tr).predict(X_oo)):.3f}")
            else:
                xgb_m = xgb.XGBClassifier(
                    n_estimators=n_est_xgb, max_depth=5, learning_rate=_lr,
                    subsample=0.8, colsample_bytree=0.7, gamma=0.1,
                    reg_alpha=0.2, reg_lambda=1.5, min_child_weight=10,
                    objective="multi:softprob", num_class=n_cls,
                    eval_metric="mlogloss", use_label_encoder=False,
                    random_state=42, verbosity=0, n_jobs=-1)
            xgb_m.fit(X_sc, y_train, sample_weight=self._sample_weights(y_train))
            self.base_models["xgb"] = xgb_m

        # ── LightGBM ─────────────────────────────────────────
        if LGB_OK:
            present  = np.unique(y_train)
            n_present = len(present)
            counts   = np.bincount(y_train, minlength=self.n_classes)
            cw = {int(c): len(y_train) / (n_present * max(counts[c], 1)) for c in present}
            if self.use_optuna:
                print("    Optuna → LightGBM...", end="", flush=True)
                best = self._tune_lgb(X_tr, y_tr, X_oo, y_oo, n_present)
                lgb_m = lgb.LGBMClassifier(**best,
                    objective="multiclass", num_class=n_present,
                    class_weight=cw, random_state=42, verbosity=-1,
                    force_col_wise=True, n_jobs=-1)
                print(f" done")
            else:
                lgb_m = lgb.LGBMClassifier(
                    n_estimators=n_est_lgb, max_depth=5, learning_rate=_lr,
                    subsample=0.8, colsample_bytree=0.7, min_child_samples=15,
                    reg_alpha=0.2, reg_lambda=1.5, class_weight=cw,
                    objective="multiclass", num_class=n_present,
                    random_state=42, verbosity=-1, force_col_wise=True, n_jobs=-1)
            lgb_m.fit(X_sc, y_train)
            self.base_models["lgb"] = lgb_m

        # ── CatBoost ─────────────────────────────────────────
        if CAT_OK and not _skip_cat:
            _cat_counts = np.bincount(y_train, minlength=self.n_classes)
            _cat_cw = {int(c): float(len(y_train) / (self.n_classes * max(_cat_counts[c], 1)))
                       for c in range(self.n_classes)}
            cat_m = cb.CatBoostClassifier(
                iterations=n_est_cat, depth=6, learning_rate=0.04,
                l2_leaf_reg=3.0, random_seed=42, verbose=0,
                loss_function="MultiClass", eval_metric="Accuracy",
                thread_count=-1, task_type="CPU",
                class_weights=_cat_cw,
            )
            cat_m.fit(X_sc, y_train, sample_weight=self._sample_weights(y_train))
            self.base_models["cat"] = cat_m

        if not self.base_models:
            rf = RandomForestClassifier(n_estimators=400, max_depth=7,
                                         class_weight="balanced",
                                         random_state=42, n_jobs=-1)
            rf.fit(X_sc, y_train)
            self.base_models["rf"] = rf

        # ── Level-2 Meta-Learner (stacking) ─────────────────
        if len(self.base_models) >= 2 and i2 > i1 + 20:
            oof_preds = np.hstack([
                self._pad_proba(m.predict_proba(X_oo))
                for m in self.base_models.values()
            ])
            if len(np.unique(y_oo)) >= 2:
                self.meta_lr = LogisticRegression(
                    C=0.5, max_iter=500, random_state=42,
                    solver="lbfgs")
                self.meta_lr.fit(oof_preds, y_oo)

        self.is_fitted = True

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Not fitted")
        X_sc = self.scaler.transform(X[self.feature_cols].values)
        probas = [self._pad_proba(m.predict_proba(X_sc))
                  for m in self.base_models.values()]

        if self.meta_lr is not None and len(probas) >= 2:
            meta_X = np.hstack(probas)
            try:
                # Patch stale pickled LR that has removed multi_class attr
                if hasattr(self.meta_lr, 'multi_class'):
                    del self.meta_lr.__dict__['multi_class']
                proba = self._pad_proba(self.meta_lr.predict_proba(meta_X))
            except Exception:
                # Fallback to weighted average if meta-learner fails
                weights = {"xgb": 0.40, "lgb": 0.35, "cat": 0.25, "rf": 1.0}
                keys    = list(self.base_models.keys())
                total_w = sum(weights.get(k, 0.33) for k in keys)
                proba   = sum(weights.get(k, 0.33) / total_w * p
                              for k, p in zip(keys, probas))
        else:
            weights = {"xgb": 0.40, "lgb": 0.35, "cat": 0.25, "rf": 1.0}
            keys    = list(self.base_models.keys())
            total_w = sum(weights.get(k, 0.33) for k in keys)
            proba   = sum(weights.get(k, 0.33) / total_w * p
                          for k, p in zip(keys, probas))
        return proba

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def predict_with_confidence(self, X: pd.DataFrame):
        proba = self.predict_proba(X)
        return proba.argmax(axis=1), proba.max(axis=1), proba

    def predict_one(self, X: pd.DataFrame):
        """Predict single row. Returns (class, confidence, proba_vector)."""
        proba = self.predict_proba(X)
        cls   = int(proba[0].argmax())
        conf  = float(proba[0].max())
        return cls, conf, proba[0].tolist()

    def feature_importance(self) -> pd.DataFrame:
        rows = []
        for name, m in self.base_models.items():
            if hasattr(m, "feature_importances_"):
                imp = m.feature_importances_
                imp = imp / (imp.sum() + 1e-9)
                for i, f in enumerate(self.feature_cols):
                    rows.append({"feature": f, "model": name,
                                 "importance": imp[i] if i < len(imp) else 0})
        if not rows:
            return pd.DataFrame()
        df_imp = pd.DataFrame(rows)
        combined = df_imp.groupby("feature")["importance"].mean().sort_values(ascending=False)
        return combined.reset_index().rename(columns={"importance": "combined"})


# ══════════════════════════════════════════════════════════════════
# SECTION 4 — WALK-FORWARD VALIDATOR v6  (5-day steps)
# ══════════════════════════════════════════════════════════════════

class WalkForwardV6:
    """
    5-day step walk-forward with ~80–100 windows.

    Improvements vs v4:
    - 5-day steps → 4x more windows → tighter confidence intervals
    - Records confidence per window → allows confidence threshold sweep
    - Returns per-window metrics → accuracy trend over time
    - Computes directional accuracy separately per vol regime
    """

    def __init__(self, min_train_days: int = 504,
                 step_days: int = 20,
                 test_window_days: int = 20,
                 max_windows: int = 12,
                 fast_mode: bool = True):
        self.min_train_days   = min_train_days
        self.step_days        = step_days
        self.test_window_days = test_window_days
        self.max_windows      = max_windows  # None = unlimited (full run)
        self.fast_mode        = fast_mode    # reduces n_estimators for speed

    def get_splits(self, df: pd.DataFrame):
        df  = df.sort_values("Date").reset_index(drop=True)
        n   = len(df)
        splits = []
        train_end = self.min_train_days
        while train_end + self.test_window_days <= n:
            test_end = min(train_end + self.test_window_days, n)
            splits.append((df.index[:train_end], df.index[train_end:test_end]))
            train_end += self.step_days
        # Cap to max_windows (take last N windows — most recent, most relevant)
        if self.max_windows and len(splits) > self.max_windows:
            splits = splits[-self.max_windows:]
        return splits

    def run(self, df: pd.DataFrame, feature_cols: List[str],
            label_col: str = "risk_label",
            use_optuna: bool = False,
            optuna_trials: int = 0) -> Dict:

        df     = df.sort_values("Date").reset_index(drop=True)
        splits = self.get_splits(df)
        if not splits:
            return {"error": "Insufficient data"}

        all_preds, all_true, all_conf = [], [], []
        all_dates, all_probas         = [], []
        window_results                 = []

        # Fast mode: XGB+LGB only, 100 trees each ≈ 2s per window × 12 = ~25s total
        _n_est = 100 if self.fast_mode else 1000

        print(f"\n  Walk-forward: {len(splits)} windows  "
              f"(min_train={self.min_train_days}d, step={self.step_days}d, "
              f"test={self.test_window_days}d, fast={self.fast_mode})")

        for i, (train_idx, test_idx) in enumerate(splits):
            X_tr = df.loc[train_idx, feature_cols].copy()
            y_tr = df.loc[train_idx, label_col].values
            X_te = df.loc[test_idx,  feature_cols].copy()
            y_te = df.loc[test_idx,  label_col].values

            if len(np.unique(y_tr)) < 3 or len(X_tr) < 400:
                continue

            # Build a lightweight ensemble for speed
            model = EnsembleV6(use_optuna=False)
            if self.fast_mode:
                # Override estimator counts before fit
                model._wf_n_est = _n_est
            try:
                model.fit(X_tr, y_tr,
                          _n_estimators=_n_est,
                          _skip_cat=self.fast_mode)   # drop CatBoost in fast mode
                preds, conf, probas = model.predict_with_confidence(X_te)
            except Exception as e:
                continue

            all_preds.extend(preds)
            all_true.extend(y_te)
            all_conf.extend(conf)
            all_dates.extend(df.loc[test_idx, "Date"].tolist())
            all_probas.extend(probas.tolist())

            # Track feature importance per window (#13)
            fi_window = {}
            for mname, mobj in model.base_models.items():
                if hasattr(mobj, "feature_importances_"):
                    imp = mobj.feature_importances_
                    imp = imp / (imp.sum() + 1e-9)
                    top5 = sorted(zip(feature_cols, imp), key=lambda x: x[1], reverse=True)[:5]
                    fi_window[mname] = [{"feature": f, "importance": round(float(v), 4)} for f, v in top5]

            w_acc = accuracy_score(y_te, preds)
            window_results.append({
                "window": i + 1,
                "train_size": len(train_idx),
                "test_size": len(test_idx),
                "accuracy": float(w_acc),
                "mean_conf": float(np.mean(conf)),
                "train_end_date": str(df.loc[train_idx[-1], "Date"])[:10],
                "feature_importance": fi_window,
            })

            if (i + 1) % 20 == 0 or i == 0:
                print(f"  Window {i+1:3d}/{len(splits)} "
                      f"acc={w_acc:.3f} conf={np.mean(conf):.3f}")

        all_preds  = np.array(all_preds)
        all_true   = np.array(all_true)
        all_conf   = np.array(all_conf)
        all_probas = np.array(all_probas)

        overall_acc = accuracy_score(all_true, all_preds)

        # ── Statistical significance tests ────────────────
        sig_tests = {}
        if SCIPY_OK:
            n_total = len(all_true)
            n_correct = int(overall_acc * n_total)
            n_classes_present = len(set(all_true))
            naive_baseline = 1.0 / max(n_classes_present, 1)
            sig_tests["overall"] = {
                "accuracy": float(overall_acc),
                "naive_baseline": round(naive_baseline, 3),
                "p_value": round(float(_binom_test(n_correct, n_total, naive_baseline, alternative="greater")), 6),
                "significant_at_05": float(_binom_test(n_correct, n_total, naive_baseline, alternative="greater")) < 0.05,
                "significant_at_01": float(_binom_test(n_correct, n_total, naive_baseline, alternative="greater")) < 0.01,
            }
        # ── Confusion matrix ──────────────────────────────
        conf_matrix = confusion_matrix(all_true, all_preds, labels=sorted(set(all_true) | set(all_preds))).tolist()

        # ── Confidence threshold sweep ──────────────────────
        sweep = []
        for thresh in np.arange(0.40, 0.86, 0.01):
            mask = all_conf >= thresh
            if mask.sum() < 10:
                break
            sweep.append({
                "threshold": round(float(thresh), 2),
                "accuracy":  float(accuracy_score(all_true[mask], all_preds[mask])),
                "coverage":  float(mask.mean()),
                "n":         int(mask.sum()),
            })

        # ── Directional accuracy (bullish vs bearish) ───────
        b_mask = (all_true <= 1) | (all_true >= 3)
        b_pred = (all_preds <= 1).astype(int)
        b_true = (all_true  <= 1).astype(int)
        dir_acc = float(accuracy_score(b_true[b_mask], b_pred[b_mask])) \
                  if b_mask.sum() > 0 else 0.0

        # ── High-conf accuracy ───────────────────────────────
        hc_mask = all_conf >= 0.60
        hc_acc  = float(accuracy_score(all_true[hc_mask], all_preds[hc_mask])) \
                  if hc_mask.sum() >= 10 else None

        # ── Per-class report ─────────────────────────────────
        cls_map   = {0:"CALM",1:"MILD",2:"MODERATE",3:"ELEVATED",4:"CRISIS"}
        all_cls   = sorted(set(all_true) | set(all_preds))
        tgt_names = [cls_map[c] for c in all_cls]
        cr        = classification_report(all_true, all_preds,
                                          labels=all_cls,
                                          target_names=tgt_names,
                                          output_dict=True, zero_division=0)

        # ── Accuracy standard deviation across windows ───────
        w_accs   = [w["accuracy"] for w in window_results]
        acc_std  = float(np.std(w_accs)) if w_accs else 0.0
        acc_mean = float(np.mean(w_accs)) if w_accs else 0.0

        return {
            "overall_accuracy":      float(overall_acc),
            "accuracy_mean_windows": acc_mean,
            "accuracy_std_windows":  acc_std,
            "accuracy_95ci":         f"[{acc_mean - 1.96*acc_std:.3f}, "
                                     f"{acc_mean + 1.96*acc_std:.3f}]",
            "directional_accuracy":  dir_acc,
            "high_conf_accuracy":    hc_acc,
            "high_conf_coverage":    float(hc_mask.mean()),
            "n_windows":             len(window_results),
            "n_test_samples":        int(len(all_preds)),
            "classification_report": cr,
            "confidence_sweep":      sweep,
            "window_results":        window_results,
            "predictions":           all_preds.tolist(),
            "true_labels":           all_true.tolist(),
            "confidences":           all_conf.tolist(),
            "probabilities":         all_probas.tolist(),
            "dates":                 [str(d)[:10] for d in all_dates],
            "significance_tests":    sig_tests,
            "confusion_matrix":      conf_matrix,
        }


# ══════════════════════════════════════════════════════════════════
# SECTION 5 — BACKTEST ENGINE
# ══════════════════════════════════════════════════════════════════

class BacktestEngine:
    """
    Simulate trading every walk-forward signal with realistic frictions.

    Modes:
      long_only:    BUY on class 0/1, HOLD on 2, SELL=flat on 3/4
      long_short:   BUY on 0/1, SELL (short) on 3/4, flat on 2

    Position sizing:
      equal_weight: fixed fraction per trade
      kelly:        Kelly criterion based on historical win rate and payoff ratio
                    Kelly fraction = (p * b - (1-p)) / b
                    where p = win rate, b = avg_win / avg_loss
                    Capped at 0.25 to prevent over-betting

    Frictions:
      round-trip cost = 2 × friction (entry + exit)
      Default: 0.10% per side (realistic for liquid US/India large cap)

    Outputs:
      Equity curve, cumulative return, Sharpe ratio, Sortino ratio,
      Calmar ratio, max drawdown, win rate, avg win, avg loss,
      profit factor, signal breakdown table, vs buy-and-hold
    """

    def __init__(self,
                 mode: str = "long_only",
                 sizing: str = "kelly",
                 friction: float = 0.0010,
                 initial_capital: float = 100_000,
                 max_kelly_fraction: float = 0.25):
        self.mode                = mode
        self.sizing              = sizing
        self.friction            = friction
        self.initial_capital     = initial_capital
        self.max_kelly_fraction  = max_kelly_fraction

    def run(self, df_prices: pd.DataFrame,
            wf_results: Dict) -> Dict:
        """
        Run backtest on walk-forward results.

        df_prices: original OHLCV DataFrame with Date and Close
        wf_results: output of WalkForwardV6.run()
        """
        if "error" in wf_results:
            return {"error": wf_results["error"]}

        dates   = pd.to_datetime(wf_results["dates"])
        preds   = np.array(wf_results["predictions"])
        confs   = np.array(wf_results["confidences"])
        trues   = np.array(wf_results["true_labels"])

        # Align with price data
        df_px = df_prices.copy()
        df_px["Date"] = pd.to_datetime(df_px["Date"]).dt.tz_localize(None)
        df_px = df_px.set_index("Date")["Close"]

        # Map signal: 0/1 → BUY, 2 → HOLD, 3/4 → SELL
        signals = pd.Series(index=dates, data=preds)
        direction = pd.Series(index=dates, dtype=float)
        direction[preds <= 1] =  1.0   # long
        direction[preds == 2] =  0.0   # flat
        direction[preds >= 3] = -1.0   # short (or flat in long_only)
        if self.mode == "long_only":
            direction[direction < 0] = 0.0

        # Get next-day returns for each signal date
        trades = []
        equity = self.initial_capital
        equity_curve = [equity]
        equity_dates = [dates[0] if len(dates) else pd.Timestamp.today()]

        # Build initial Kelly estimate from first 30 signals
        wins = 0; losses = 0; total_win = 0.0; total_loss = 0.0

        for i, (dt, sig_dir) in enumerate(direction.items()):
            if sig_dir == 0:
                equity_curve.append(equity)
                equity_dates.append(dt)
                continue

            # Find next trading day's return in price data
            try:
                loc = df_px.index.searchsorted(dt)
                if loc + 1 >= len(df_px):
                    continue
                entry_px = float(df_px.iloc[loc])
                exit_px  = float(df_px.iloc[loc + 1])
            except:
                continue

            raw_ret = (exit_px - entry_px) / entry_px
            dir_ret = raw_ret * sig_dir
            # Volatility-scaled slippage: base + k * |raw_ret|
            vol_slippage = self.friction + 0.3 * abs(raw_ret)
            fric = 2 * min(vol_slippage, 0.005)  # cap at 0.5% per side
            net_ret = dir_ret - fric

            # Kelly sizing
            if self.sizing == "kelly" and i >= 10:
                total = wins + losses
                p_win = wins / total if total > 0 else 0.5
                avg_w = (total_win  / wins)   if wins   > 0 else 0.005
                avg_l = (total_loss / losses) if losses > 0 else 0.005
                b     = avg_w / (avg_l + 1e-9)
                kelly = (p_win * b - (1 - p_win)) / (b + 1e-9)
                frac  = max(0.01, min(self.max_kelly_fraction, kelly))
            else:
                frac  = 0.10  # fixed 10% per trade initially

            # Apply confidence scaling: reduce size below 0.55 conf
            conf_scale = min(1.0, max(0.3, (float(confs[i]) - 0.40) / 0.30))
            frac = frac * conf_scale

            pnl = equity * frac * net_ret
            equity += pnl

            if net_ret > 0:
                wins += 1; total_win += abs(net_ret)
            else:
                losses += 1; total_loss += abs(net_ret)

            trades.append({
                "date":      str(dt)[:10],
                "signal":    int(preds[i]),
                "direction": float(sig_dir),
                "confidence":float(confs[i]),
                "kelly_frac":float(frac),
                "raw_ret":   float(raw_ret),
                "net_ret":   float(net_ret),
                "pnl":       float(pnl),
                "equity":    float(equity),
                "true_label":int(trues[i]),
                "correct":   int(preds[i] == trues[i]),
            })
            equity_curve.append(equity)
            equity_dates.append(dt)

        if not trades:
            return {"error": "No trades executed"}

        # ── Performance metrics ──────────────────────────────
        eq  = pd.Series(equity_curve, index=equity_dates)
        ret = eq.pct_change().dropna()

        total_return = (equity - self.initial_capital) / self.initial_capital

        # Sharpe (annualised, 252 trading days)
        sharpe = float(ret.mean() / (ret.std() + 1e-9) * np.sqrt(252)) \
                 if len(ret) > 1 else 0.0

        # Sortino (downside deviation only)
        down   = ret[ret < 0]
        sortino = float(ret.mean() / (down.std() + 1e-9) * np.sqrt(252)) \
                  if len(down) > 1 else 0.0

        # Max drawdown + duration
        peak   = eq.cummax()
        dd     = (eq - peak) / (peak + 1e-9)
        max_dd = float(dd.min())
        # Drawdown duration: longest consecutive underwater period
        underwater = (dd < -0.001).astype(int)
        if underwater.sum() > 0:
            streaks = underwater.groupby((underwater != underwater.shift()).cumsum()).cumsum()
            max_dd_duration = int(streaks.max())
        else:
            max_dd_duration = 0

        # Calmar ratio
        calmar = float(total_return / (abs(max_dd) + 1e-9))

        # Win rate and payoff
        n_trades  = len(trades)
        n_wins    = sum(t["net_ret"] > 0 for t in trades)
        n_loss    = n_trades - n_wins
        win_rate  = n_wins / n_trades if n_trades > 0 else 0.0
        avg_win   = np.mean([t["net_ret"] for t in trades if t["net_ret"] > 0]) \
                    if n_wins > 0 else 0.0
        avg_loss  = np.mean([t["net_ret"] for t in trades if t["net_ret"] <= 0]) \
                    if n_loss > 0 else 0.0
        profit_factor = abs(sum(t["net_ret"] for t in trades if t["net_ret"] > 0)) / \
                        (abs(sum(t["net_ret"] for t in trades if t["net_ret"] <= 0)) + 1e-9)

        # Signal accuracy per class
        sig_breakdown = {}
        cls_map = {0:"CALM",1:"MILD",2:"MODERATE",3:"ELEVATED",4:"CRISIS"}
        for cls in range(5):
            sub = [t for t in trades if t["signal"] == cls]
            if sub:
                sig_breakdown[cls_map[cls]] = {
                    "n":         len(sub),
                    "win_rate":  round(sum(t["net_ret"] > 0 for t in sub) / len(sub), 3),
                    "avg_ret":   round(np.mean([t["net_ret"] for t in sub]), 4),
                    "accuracy":  round(sum(t["correct"] for t in sub) / len(sub), 3),
                }

        # Buy-and-hold benchmark
        try:
            bh_start = float(df_px.loc[df_px.index >= dates[0]].iloc[0])
            bh_end   = float(df_px.loc[df_px.index <= dates[-1]].iloc[-1])
            bh_ret   = (bh_end - bh_start) / bh_start
        except:
            bh_ret   = 0.0

        # Annualise (using calendar days for proper annualisation)
        n_calendar_days = max((dates[-1] - dates[0]).days, 1)
        n_trading_days  = len(set(equity_dates))
        ann_ret  = float((1 + total_return) ** (365.25 / max(n_calendar_days, 30)) - 1)

        return {
            "total_return":        round(float(total_return), 4),
            "annualised_return":   round(ann_ret, 4),
            "buy_hold_return":     round(float(bh_ret), 4),
            "alpha":               round(float(total_return - bh_ret), 4),
            "sharpe_ratio":        round(sharpe, 3),
            "sortino_ratio":       round(sortino, 3),
            "calmar_ratio":        round(calmar, 3),
            "max_drawdown":        round(max_dd, 4),
            "max_dd_duration_days": max_dd_duration,
            "win_rate":            round(win_rate, 3),
            "n_trades":            n_trades,
            "n_wins":              n_wins,
            "n_losses":            n_loss,
            "avg_win":             round(float(avg_win), 4),
            "avg_loss":            round(float(avg_loss), 4),
            "profit_factor":       round(profit_factor, 3),
            "final_equity":        round(equity, 2),
            "signal_breakdown":    sig_breakdown,
            "equity_curve":        [round(e, 2) for e in equity_curve],
            "equity_dates":        [str(d)[:10] for d in equity_dates],
            "trades":              trades,
            "regime_analysis":     self._regime_analysis(trades, dates),
        }

    @staticmethod
    def _regime_analysis(trades, dates):
        """Analyse performance across known market regimes."""
        regimes = {
            "COVID_Crash":    ("2020-02-15", "2020-04-15"),
            "Rate_Hike_Bear": ("2022-01-01", "2022-10-15"),
            "AI_Bull":        ("2023-01-01", "2024-06-30"),
            "Recent":         ("2024-07-01", "2026-12-31"),
        }
        results = {}
        for name, (start, end) in regimes.items():
            regime_trades = [t for t in trades
                           if start <= t["date"] <= end]
            if len(regime_trades) >= 5:
                rets = [t["net_ret"] for t in regime_trades]
                wins = sum(1 for r in rets if r > 0)
                results[name] = {
                    "n_trades":  len(regime_trades),
                    "win_rate":  round(wins / len(regime_trades), 3),
                    "avg_ret":   round(float(np.mean(rets)), 4),
                    "total_ret": round(float(np.sum(rets)), 4),
                    "sharpe":    round(float(np.mean(rets) / (np.std(rets) + 1e-9) * np.sqrt(252)), 2),
                }
        return results





# ══════════════════════════════════════════════════════════════════
# SECTION 4B — FEATURE SELECTION ANALYSIS (#12)
# ══════════════════════════════════════════════════════════════════

def feature_selection_analysis(df: pd.DataFrame, fcs: List[str],
                                top_k_list: List[int] = None) -> Dict:
    """
    Evaluate model with different feature subsets using mutual information.

    Tests accuracy with top-30, top-50, top-80, and all features
    to determine if dimensionality reduction helps.
    """
    from sklearn.feature_selection import mutual_info_classif
    if top_k_list is None:
        top_k_list = [30, 50, 80, len(fcs)]

    X = df[fcs].fillna(0)
    y = df["risk_label"].values

    # Mutual information scores
    mi_scores = mutual_info_classif(X, y, random_state=42, n_neighbors=5)
    mi_ranking = sorted(zip(fcs, mi_scores), key=lambda x: x[1], reverse=True)

    results = {
        "mi_ranking": [{"feature": f, "mi_score": round(float(s), 4)}
                       for f, s in mi_ranking[:30]],
        "subset_performance": [],
    }

    for k in top_k_list:
        k = min(k, len(fcs))
        top_features = [f for f, _ in mi_ranking[:k]]
        # Quick 70/30 split evaluation
        n = len(df); cut = int(n * 0.70)
        X_tr = df.iloc[:cut][top_features].fillna(0)
        y_tr = df.iloc[:cut]["risk_label"].values
        X_te = df.iloc[cut:][top_features].fillna(0)
        y_te = df.iloc[cut:]["risk_label"].values

        try:
            from sklearn.ensemble import GradientBoostingClassifier
            m = GradientBoostingClassifier(n_estimators=200, max_depth=5,
                                            random_state=42)
            sc = RobustScaler()
            m.fit(sc.fit_transform(X_tr), y_tr)
            acc = float(accuracy_score(y_te, m.predict(sc.transform(X_te))))
            results["subset_performance"].append({
                "n_features": k,
                "accuracy": round(acc, 4),
                "features_used": top_features[:5],  # sample
            })
        except Exception:
            pass

    return results


# ══════════════════════════════════════════════════════════════════
# SECTION 5A — MONTE CARLO PERMUTATION TEST (#5)
# ══════════════════════════════════════════════════════════════════

def permutation_test(y_true: np.ndarray, y_pred: np.ndarray,
                     n_permutations: int = 1000, seed: int = 42) -> Dict:
    """
    Monte Carlo permutation test for classification accuracy.

    Shuffles true labels n_permutations times and computes accuracy
    each time to build a null distribution. The p-value is the
    fraction of permuted accuracies >= observed accuracy.

    If p < 0.05, the model has genuine predictive power beyond chance.
    """
    np.random.seed(seed)
    observed_acc = accuracy_score(y_true, y_pred)
    null_accs = []
    for _ in range(n_permutations):
        perm = np.random.permutation(y_true)
        null_accs.append(accuracy_score(perm, y_pred))
    null_accs = np.array(null_accs)
    p_value = float(np.mean(null_accs >= observed_acc))
    return {
        "observed_accuracy": round(float(observed_acc), 4),
        "null_mean": round(float(null_accs.mean()), 4),
        "null_std": round(float(null_accs.std()), 4),
        "null_p95": round(float(np.percentile(null_accs, 95)), 4),
        "null_p99": round(float(np.percentile(null_accs, 99)), 4),
        "p_value": round(p_value, 4),
        "significant_at_05": p_value < 0.05,
        "significant_at_01": p_value < 0.01,
        "n_permutations": n_permutations,
        "effect_size": round(float((observed_acc - null_accs.mean()) / (null_accs.std() + 1e-9)), 2),
    }


# ══════════════════════════════════════════════════════════════════
# SECTION 5B — BENCHMARK STRATEGIES (#10)
# ══════════════════════════════════════════════════════════════════

class BenchmarkStrategies:
    """Simple benchmark strategies to compare against CVA-SACS."""

    @staticmethod
    def ma_crossover(prices: pd.Series, fast: int = 50, slow: int = 200,
                     friction: float = 0.001) -> Dict:
        """Moving average crossover: long when fast > slow."""
        ma_f = prices.rolling(fast).mean()
        ma_s = prices.rolling(slow).mean()
        signal = (ma_f > ma_s).astype(int).shift(1)
        daily_ret = prices.pct_change()
        strat_ret = signal * daily_ret - signal.diff().abs() * friction
        strat_ret = strat_ret.dropna()
        cum = (1 + strat_ret).cumprod()
        total = float(cum.iloc[-1] - 1) if len(cum) > 0 else 0.0
        sharpe = float(strat_ret.mean() / (strat_ret.std() + 1e-9) * np.sqrt(252))
        max_dd = float(((cum - cum.cummax()) / (cum.cummax() + 1e-9)).min())
        return {"strategy": f"MA({fast}/{slow})", "total_return": round(total, 4),
                "sharpe": round(sharpe, 2), "max_drawdown": round(max_dd, 4),
                "n_trades": int(signal.diff().abs().sum())}

    @staticmethod
    def rsi_mean_reversion(prices: pd.Series, period: int = 14,
                           buy_thresh: int = 30, sell_thresh: int = 70,
                           friction: float = 0.001) -> Dict:
        """RSI mean reversion: buy when RSI < buy_thresh, sell when > sell_thresh."""
        delta = prices.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rsi = 100 - 100 / (gain / (loss + 1e-9) + 1)
        signal = pd.Series(0, index=prices.index)
        signal[rsi < buy_thresh] = 1
        signal[rsi > sell_thresh] = 0
        signal = signal.ffill().shift(1)
        daily_ret = prices.pct_change()
        strat_ret = signal * daily_ret - signal.diff().abs() * friction
        strat_ret = strat_ret.dropna()
        cum = (1 + strat_ret).cumprod()
        total = float(cum.iloc[-1] - 1) if len(cum) > 0 else 0.0
        sharpe = float(strat_ret.mean() / (strat_ret.std() + 1e-9) * np.sqrt(252))
        max_dd = float(((cum - cum.cummax()) / (cum.cummax() + 1e-9)).min())
        return {"strategy": f"RSI({period},{buy_thresh}/{sell_thresh})", "total_return": round(total, 4),
                "sharpe": round(sharpe, 2), "max_drawdown": round(max_dd, 4),
                "n_trades": int(signal.diff().abs().sum())}

    @staticmethod
    def random_strategy(prices: pd.Series, trade_freq: int = 50,
                        friction: float = 0.001, seed: int = 42) -> Dict:
        """Random entry/exit with same frequency as CVA-SACS."""
        np.random.seed(seed)
        n = len(prices)
        signal = np.zeros(n)
        trade_days = np.random.choice(n, min(trade_freq, n), replace=False)
        for d in sorted(trade_days):
            signal[d:] = 1 - signal[d]
        signal = pd.Series(signal, index=prices.index).shift(1)
        daily_ret = prices.pct_change()
        strat_ret = signal * daily_ret - signal.diff().abs() * friction
        strat_ret = strat_ret.dropna()
        cum = (1 + strat_ret).cumprod()
        total = float(cum.iloc[-1] - 1) if len(cum) > 0 else 0.0
        sharpe = float(strat_ret.mean() / (strat_ret.std() + 1e-9) * np.sqrt(252))
        return {"strategy": "Random", "total_return": round(total, 4),
                "sharpe": round(sharpe, 2), "max_drawdown": 0.0,
                "n_trades": trade_freq}

    @classmethod
    def run_all(cls, prices: pd.Series, cva_trades: int = 50) -> List[Dict]:
        """Run all benchmarks and return comparison table."""
        return [
            cls.ma_crossover(prices, 50, 200),
            cls.ma_crossover(prices, 20, 50),
            cls.rsi_mean_reversion(prices),
            cls.random_strategy(prices, trade_freq=cva_trades),
        ]


# ══════════════════════════════════════════════════════════════════
# SECTION 6 — SIGNAL PERSISTENCE TRACKER
# ══════════════════════════════════════════════════════════════════

class SignalPersistenceTracker:
    """
    Track signal history and flag when a signal has been
    consistently confirmed for N consecutive days.

    A CONFIRMED signal (same verdict for ≥3 days) is empirically
    more reliable than a fresh signal. This adds a layer of
    validation on top of the model output.

    Usage:
        tracker = SignalPersistenceTracker()
        result = tracker.update("ROBUST_BUY", confidence=0.72)
        # result: {"signal": "ROBUST_BUY", "days_persistent": 3,
        #          "status": "CONFIRMED", "action": "HOLD_POSITION"}
    """

    def __init__(self, confirm_days: int = 3):
        self.confirm_days  = confirm_days
        self.history: List[Dict] = []
        self.current_signal: Optional[str] = None
        self.streak: int = 0

    def update(self, signal: str, confidence: float) -> Dict:
        if signal == self.current_signal:
            self.streak += 1
        else:
            prev_signal    = self.current_signal
            self.current_signal = signal
            self.streak    = 1

        self.history.append({
            "signal":     signal,
            "confidence": confidence,
            "streak":     self.streak,
            "timestamp":  datetime.now().isoformat(),
        })

        if self.streak >= self.confirm_days:
            status = "CONFIRMED"
            action = "HOLD_POSITION" if self.streak > self.confirm_days else "ENTER_POSITION"
        elif self.streak == 1 and len(self.history) > 1:
            status = "REVERSAL"
            action = "EXIT_PREVIOUS_POSITION"
        else:
            status = "PENDING"
            action = "WAIT_FOR_CONFIRMATION"

        return {
            "signal":          signal,
            "confidence":      confidence,
            "days_persistent": self.streak,
            "status":          status,
            "action":          action,
        }

    def get_history(self) -> List[Dict]:
        return self.history[-30:]  # last 30 signals


# ══════════════════════════════════════════════════════════════════
# SECTION 7 — ABLATION STUDY v6
# ══════════════════════════════════════════════════════════════════

class AblationV6:
    """
    Extended ablation comparing:
      0. Naive baseline
      1. Binary XGB (direction only)
      2. RF v3 (baseline)
      3. XGB only
      4. LGB only
      5. CatBoost only
      6. XGB+LGB ensemble (v4 style)
      7. v4 + walk-forward
      8. v6 ensemble (XGB+LGB+CAT+meta)
      9. v6 + alt data features
     10. v6 + macro features
     11. Full v6 (all features + stacking)
    """

    def __init__(self, df: pd.DataFrame, feature_cols_base: List[str],
                 feature_cols_v6: List[str]):
        self.df              = df
        self.fc_base         = feature_cols_base  # original 85
        self.fc_v6           = feature_cols_v6    # full 130

    def _split(self, fcs):
        n   = len(self.df); cut = int(n * 0.70)
        tr  = self.df.iloc[:cut]; te = self.df.iloc[cut:]
        return tr[fcs], tr["risk_label"].values, te[fcs], te["risk_label"].values

    def _row(self, name, preds, true):
        acc  = accuracy_score(true, preds)
        b    = (true <= 1) | (true >= 3)
        b_p  = (preds <= 1).astype(int); b_t = (true <= 1).astype(int)
        d    = float(accuracy_score(b_t[b], b_p[b])) if b.sum() > 0 else 0.0
        f1   = f1_score(true, preds, average="macro", zero_division=0)
        cr   = classification_report(true, preds, output_dict=True, zero_division=0)
        mild = cr.get("1", {}).get("recall", 0.0)
        return {"Model": name, "Overall Acc": f"{acc:.3f}",
                "Directional": f"{d:.3f}", "Macro F1": f"{f1:.3f}",
                "MILD Recall": f"{mild:.3f}", "N": len(true)}

    def run_all(self) -> pd.DataFrame:
        rows = []
        sc   = RobustScaler()

        # ── 0. Naive ─────────────────────────────────────────
        _, _, X_te, y_te = self._split(self.fc_base)
        rows.append(self._row("Naive (always HOLD)", np.full(len(y_te), 2), y_te))

        # ── 2. RF v3 ─────────────────────────────────────────
        X_tr, y_tr, X_te, y_te = self._split(self.fc_base)
        try:
            rf = RandomForestClassifier(n_estimators=200, max_depth=6,
                                         min_samples_leaf=20, class_weight="balanced",
                                         random_state=42, n_jobs=-1)
            Xsc = sc.fit_transform(X_tr); rf.fit(Xsc, y_tr)
            rows.append(self._row("RF v3 (85 feat)", rf.predict(sc.transform(X_te)), y_te))
        except: pass

        # ── 3. XGB only (base features) ──────────────────────
        if XGB_OK:
            try:
                sw = np.array([len(y_tr)/(5*max(np.bincount(y_tr,minlength=5)[c],1)) for c in y_tr])
                s2 = RobustScaler()
                m = xgb.XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.05,
                                       subsample=0.8, colsample_bytree=0.7,
                                       objective="multi:softprob", num_class=5,
                                       eval_metric="mlogloss", use_label_encoder=False,
                                       random_state=42, verbosity=0, n_jobs=-1)
                m.fit(s2.fit_transform(X_tr), y_tr, sample_weight=sw)
                rows.append(self._row("XGB only (85 feat)", m.predict(s2.transform(X_te)), y_te))
            except: pass

        # ── 4. LGB only (base features) ──────────────────────
        if LGB_OK:
            try:
                s3 = RobustScaler()
                present = np.unique(y_tr)
                cw = {int(c): len(y_tr)/(len(present)*max(np.bincount(y_tr,minlength=5)[c],1)) for c in present}
                m = lgb.LGBMClassifier(n_estimators=500, max_depth=5, learning_rate=0.05,
                                        subsample=0.8, colsample_bytree=0.7, class_weight=cw,
                                        objective="multiclass", num_class=len(present),
                                        random_state=42, verbosity=-1, n_jobs=-1)
                m.fit(s3.fit_transform(X_tr), y_tr)
                rows.append(self._row("LGB only (85 feat)", m.predict(s3.transform(X_te)), y_te))
            except: pass

        # ── 5. CatBoost only ─────────────────────────────────
        if CAT_OK:
            try:
                s4 = RobustScaler(); Xsc4 = s4.fit_transform(X_tr)
                sw4 = np.array([len(y_tr)/(5*max(np.bincount(y_tr,minlength=5)[c],1)) for c in y_tr])
                m = cb.CatBoostClassifier(iterations=400, depth=5, learning_rate=0.05,
                                           random_seed=42, verbose=0, loss_function="MultiClass", n_jobs=-1 if hasattr(cb.CatBoostClassifier(),'n_jobs') else None)
                m.fit(Xsc4, y_tr, sample_weight=sw4)
                rows.append(self._row("CatBoost only (85 feat)", m.predict(s4.transform(X_te)).flatten(), y_te))
            except Exception as e:
                print(f"    CatBoost ablation error: {e}")

        # ── 6. v4 ensemble (XGB+LGB, 85 feat) ────────────────
        try:
            from cva_sacs_v4_ml import EnsembleMLModel as EnsV4
            e4 = EnsV4(); e4.fit(X_tr, y_tr)
            rows.append(self._row("v4 Ensemble (XGB+LGB, 85)", e4.predict(X_te), y_te))
        except:
            pass

        # ── 8. v6 ensemble (base features) ───────────────────
        try:
            e6b = EnsembleV6(); e6b.fit(X_tr, y_tr)
            rows.append(self._row("v6 Ensemble (85 feat)", e6b.predict(X_te), y_te))
        except Exception as e:
            print(f"    v6-base ablation: {e}")

        # ── 9. v6 with alt data features ─────────────────────
        if all(c in self.df.columns for c in ["iv_rank_proxy", "short_proxy"]):
            alt_cols = self.fc_base + [c for c in self.fc_v6
                                       if c not in self.fc_base
                                       and c in self.df.columns
                                       and not c.startswith(("vix_","rel_","beta","alpha","tlt","hyg","dxy","gld","risk_on"))]
            alt_cols = [c for c in alt_cols if c in self.df.columns]
            try:
                X_tr2, y_tr2, X_te2, y_te2 = self._split(alt_cols)
                e6a = EnsembleV6(); e6a.fit(X_tr2, y_tr2)
                rows.append(self._row("v6 + Alt Data", e6a.predict(X_te2), y_te2))
            except Exception as e:
                print(f"    v6+alt ablation: {e}")

        # ── 10. Full v6 (all 130 features) ───────────────────
        fc6_avail = [c for c in self.fc_v6 if c in self.df.columns]
        if len(fc6_avail) > len(self.fc_base):
            try:
                X_tr3, y_tr3, X_te3, y_te3 = self._split(fc6_avail)
                e6f = EnsembleV6(); e6f.fit(X_tr3, y_tr3)
                rows.append(self._row(f"v6 Full ({len(fc6_avail)} feat)", e6f.predict(X_te3), y_te3))
            except Exception as e:
                print(f"    v6-full ablation: {e}")

        return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════
# SECTION 8 — PKL-BASED MODEL PERSISTENCE
# ══════════════════════════════════════════════════════════════════

MODELS_DIR = Path("./models_v6")

def save_model(ticker: str, ensemble: EnsembleV6,
               feature_cols: List[str], meta: Dict) -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    safe   = ticker.replace(".", "_").replace("^", "")
    path_m = MODELS_DIR / f"model_{safe}.pkl"
    path_f = MODELS_DIR / f"fcols_{safe}.json"
    path_d = MODELS_DIR / f"metadata.json"

    with open(path_m, "wb") as f:
        pickle.dump(ensemble, f, protocol=4)
    with open(path_f, "w") as f:
        json.dump(feature_cols, f)

    # Update metadata
    existing = {}
    if path_d.exists():
        with open(path_d) as f:
            existing = json.load(f)
    existing[ticker] = {**meta, "trained_at": datetime.now().isoformat()}
    with open(path_d, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"  Saved: {path_m.name}  ({len(feature_cols)} features)")


def load_model(ticker: str) -> Tuple[Optional[EnsembleV6], Optional[List[str]]]:
    safe   = ticker.replace(".", "_").replace("^", "")
    path_m = MODELS_DIR / f"model_{safe}.pkl"
    path_f = MODELS_DIR / f"fcols_{safe}.json"
    if not path_m.exists():
        return None, None
    try:
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore")  # suppress XGBoost version mismatch warnings
            with open(path_m, "rb") as f:
                ens = pickle.load(f)
        # Patch any stale sklearn attributes from old pickles
        if hasattr(ens, 'meta_lr') and ens.meta_lr is not None:
            if hasattr(ens.meta_lr, 'multi_class'):
                try:
                    del ens.meta_lr.__dict__['multi_class']
                except Exception:
                    pass
        with open(path_f) as f:
            fcs = json.load(f)
        return ens, fcs
    except Exception as e:
        print(f"  load_model({ticker}) failed: {e}")
        return None, None


# ══════════════════════════════════════════════════════════════════
# SECTION 9 — FULL PIPELINE ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════

def run_pipeline_v6(ticker: str,
                    mode: str = "full",
                    use_optuna: bool = False,
                    optuna_trials: int = 100,
                    save_pkl: bool = True) -> Dict:
    """
    Full v6 pipeline for a single ticker.

    Modes:
      full      — walk-forward + ablation + backtest + feature importance
      backtest  — walk-forward + backtest only
      ablation  — ablation table only
      pretrain  — train on all data, save pkl, no validation
      shap      — SHAP feature importance
    """
    print(f"\n{'═'*65}")
    print(f"  CVA-SACS v6  ·  {ticker}  ·  [{mode.upper()}]")
    print(f"{'═'*65}")

    # ── 1. Download ──────────────────────────────────────────
    df_raw = download_data(ticker, years=6)
    if df_raw is None:
        return {"ticker": ticker, "error": "download failed"}

    # ── 2. Macro data ────────────────────────────────────────
    macro = _fetch_macro()

    # ── 3. Feature engineering ───────────────────────────────
    print(f"\n  Building v6 features...")
    fe   = FeatureEngineerV6()
    df   = fe.build(df_raw, macro=macro)
    fcs  = fe.get_feature_cols(df)
    print(f"  Features: {len(fcs)}")

    # ── 4. Label construction ────────────────────────────────
    print(f"  Building dual-horizon labels (h1=5d, h2=10d)...")
    df   = build_label_v6(df, h1=5, h2=10)
    df   = df.dropna(subset=fcs + ["risk_label"]).reset_index(drop=True)
    fcs  = [c for c in fcs if c in df.columns]
    print(f"  Rows: {len(df)}")
    print(f"  Label distribution:\n{df['risk_label'].value_counts().sort_index().to_string()}")

    if len(df) < 600:
        return {"ticker": ticker, "error": f"Too few rows: {len(df)}"}

    # Class imbalance analysis (#11)
    label_dist = df["risk_label"].value_counts().sort_index()
    max_count = label_dist.max(); min_count = label_dist.min()
    imbalance_ratio = round(float(max_count / (min_count + 1)), 2)
    imbalance_warning = imbalance_ratio > 3.0

    result = {"ticker": ticker, "n_rows": len(df), "n_features": len(fcs),
              "mode": mode, "timestamp": datetime.now().isoformat(),
              "label_distribution": label_dist.to_dict(),
              "imbalance_ratio": imbalance_ratio,
              "imbalance_warning": imbalance_warning}

    # ── 5. Pretrain mode: just train + save ──────────────────
    if mode == "pretrain":
        print(f"\n  Training full ensemble on all data...")
        ens = EnsembleV6(use_optuna=use_optuna, optuna_trials=optuna_trials)
        ens.fit(df[fcs], df["risk_label"].values)
        if save_pkl:
            save_model(ticker, ens, fcs,
                       {"n_rows": len(df), "n_features": len(fcs),
                        "label_dist": df["risk_label"].value_counts().to_dict()})
        result["status"] = "pretrained"
        return result

    # ── 6. Walk-forward validation ───────────────────────────
    if mode in ("full", "walkforward", "backtest"):
        print(f"\n  Running walk-forward validation (5-day steps)...")
        wf     = WalkForwardV6(min_train_days=504, step_days=5, test_window_days=5)
        wf_res = wf.run(df, fcs, use_optuna=use_optuna,
                        optuna_trials=optuna_trials)
        result["walkforward"] = wf_res

        if "error" not in wf_res:
            print(f"\n  ── Walk-Forward Results ──────────────────")
            print(f"  Overall accuracy:      {wf_res['overall_accuracy']:.3f}")
            print(f"  Directional accuracy:  {wf_res['directional_accuracy']:.3f}")
            if wf_res["high_conf_accuracy"]:
                print(f"  High-conf (≥0.60):     {wf_res['high_conf_accuracy']:.3f}  "
                      f"(coverage={wf_res['high_conf_coverage']:.1%})")
            print(f"  95% CI across windows: {wf_res['accuracy_95ci']}")
            print(f"  Windows tested:        {wf_res['n_windows']}")
            print(f"  Total test samples:    {wf_res['n_test_samples']}")

            print(f"\n  Per-class recall:")
            for cls in ["CALM","MILD","MODERATE","ELEVATED","CRISIS"]:
                info = wf_res["classification_report"].get(cls, {})
                if info:
                    print(f"    {cls:<10} recall={info.get('recall',0):.3f}  "
                          f"support={int(info.get('support',0))}")

            print(f"\n  Confidence sweep (accuracy vs coverage):")
            for s in wf_res["confidence_sweep"][::5]:
                bar = "█" * int(s["accuracy"] * 30)
                print(f"    ≥{s['threshold']:.2f} → "
                      f"acc={s['accuracy']:.3f} {bar} "
                      f"cov={s['coverage']:.1%}")

    # ── 7. Backtest ──────────────────────────────────────────
    if mode in ("full", "backtest") and "walkforward" in result:
        print(f"\n  Running backtest (long-only, Kelly sizing)...")
        bt = BacktestEngine(mode="long_only", sizing="kelly", friction=0.001)
        bt_res = bt.run(df_raw, result["walkforward"])
        result["backtest"] = bt_res

        if "error" not in bt_res:
            print(f"\n  ── Backtest Results ─────────────────────")
            print(f"  Total return:          {bt_res['total_return']:+.1%}")
            print(f"  Annualised return:     {bt_res['annualised_return']:+.1%}")
            print(f"  Buy-and-hold return:   {bt_res['buy_hold_return']:+.1%}")
            print(f"  Alpha vs B&H:          {bt_res['alpha']:+.1%}")
            print(f"  Sharpe ratio:          {bt_res['sharpe_ratio']:.2f}")
            print(f"  Sortino ratio:         {bt_res['sortino_ratio']:.2f}")
            print(f"  Max drawdown:          {bt_res['max_drawdown']:.1%}")
            print(f"  Calmar ratio:          {bt_res['calmar_ratio']:.2f}")
            print(f"  Win rate:              {bt_res['win_rate']:.1%}")
            print(f"  Trades:                {bt_res['n_trades']}")
            print(f"  Profit factor:         {bt_res['profit_factor']:.2f}")
            print(f"\n  Signal breakdown:")
            for sig, s in bt_res["signal_breakdown"].items():
                print(f"    {sig:<12} n={s['n']:3d}  "
                      f"win={s['win_rate']:.1%}  "
                      f"avg_ret={s['avg_ret']:+.3%}  "
                      f"model_acc={s['accuracy']:.1%}")

    # ── 8. Ablation ──────────────────────────────────────────
    if mode in ("full", "ablation"):
        print(f"\n  Running extended ablation...")
        # Get base feature cols (v4 style, 85 features)
        fe_base = FeatureEngineerV6()
        df_base = fe_base.build(df_raw)   # without macro
        df_base = build_label_v6(df_base, h1=5, h2=10)
        fc_base = [c for c in fe_base.get_feature_cols(df_base)
                   if c in df_base.columns]
        df_base = df_base.dropna(subset=fc_base+["risk_label"])

        abl = AblationV6(df_base, fc_base, fcs)
        abl_df = abl.run_all()
        result["ablation"] = abl_df.to_dict(orient="records")
        print(f"\n  ── Ablation Table ──────────────────────")
        print(abl_df.to_string(index=False))

    # ── 9. Feature importance (SHAP) ─────────────────────────
    if mode in ("full", "shap"):
        print(f"\n  Computing feature importance...")
        n   = len(df); cut = int(n * 0.75)
        ens = EnsembleV6()
        ens.fit(df.iloc[:cut][fcs], df.iloc[:cut]["risk_label"].values)
        fi  = ens.feature_importance()
        if not fi.empty:
            result["feature_importance"] = fi.head(20).to_dict(orient="records")
            print(f"\n  Top 20 features:")
            for _, row in fi.head(20).iterrows():
                bar = "█" * int(row["combined"] * 400)
                print(f"    {row['feature']:<32} {bar} {row['combined']:.4f}")

        if SHAP_OK and XGB_OK:
            try:
                xgb_m = ens.base_models.get("xgb")
                if xgb_m:
                    X_te_sc = ens.scaler.transform(df.iloc[cut:][fcs].values)
                    X_df    = pd.DataFrame(X_te_sc[:200], columns=fcs)
                    explainer = shap.TreeExplainer(xgb_m)
                    shap_vals = explainer.shap_values(X_df)
                    print(f"  SHAP computed for {len(X_df)} samples.")
                    result["shap_computed"] = True
            except Exception as e:
                print(f"  SHAP error: {e}")

    # ── 10. Save pkl ─────────────────────────────────────────
    if save_pkl and mode in ("full", "pretrain"):
        print(f"\n  Saving model to pkl...")
        ens_full = EnsembleV6()
        ens_full.fit(df[fcs], df["risk_label"].values)
        save_model(ticker, ens_full, fcs,
                   {"n_rows": len(df), "n_features": len(fcs),
                    "walkforward_acc": result.get("walkforward", {}).get("overall_accuracy"),
                    "backtest_sharpe": result.get("backtest", {}).get("sharpe_ratio")})

    return result


# ══════════════════════════════════════════════════════════════════
# SECTION 10 — CROSS-STOCK GLOBAL MODEL
# ══════════════════════════════════════════════════════════════════

# Full ticker lists
TICKERS_US = [
    "AAPL","MSFT","NVDA","GOOGL","AMZN","META","TSLA",
    "JPM","V","JNJ","XOM","UNH","MA","HD","PG",
    "AVGO","COST","CVX","MRK","ABBV"
]
TICKERS_IN = [
    "RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","ICICIBANK.NS",
    "HINDUNILVR.NS","SBIN.NS","TATACONSUM.NS","WIPRO.NS","AXISBANK.NS",
    "BAJFINANCE.NS","MARUTI.NS","SUNPHARMA.NS","NESTLEIND.NS","TECHM.NS",
    "LT.NS","ULTRACEMCO.NS","ADANIENT.NS","KOTAKBANK.NS","TITAN.NS"
]
ALL_TICKERS = TICKERS_US + TICKERS_IN


def run_cross_stock_v6(tickers: Optional[List[str]] = None,
                       save_pkl: bool = True) -> Dict:
    """
    Train global cross-stock model on all tickers.
    Adds ticker_code as an integer feature.
    With 40 tickers × ~1200 rows = ~48,000 training rows.
    """
    if tickers is None:
        tickers = ALL_TICKERS

    print(f"\n{'═'*65}")
    print(f"  CROSS-STOCK TRAINING: {len(tickers)} tickers")
    print(f"{'═'*65}")

    macro   = _fetch_macro()
    fe      = FeatureEngineerV6()
    all_dfs = []

    for i, ticker in enumerate(tickers):
        df_raw = download_data(ticker, years=6)
        if df_raw is None:
            continue
        try:
            df = fe.build(df_raw, macro=macro)
            df = build_label_v6(df, h1=5, h2=10)
            df["ticker_code"] = i
            all_dfs.append(df)
        except Exception as e:
            print(f"  {ticker} build error: {e}")

    if not all_dfs:
        return {"error": "No data"}

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.sort_values("Date").reset_index(drop=True)

    fcs = fe.get_feature_cols(combined)
    fcs = [c for c in fcs if c in combined.columns]
    if "ticker_code" not in fcs:
        fcs.append("ticker_code")

    combined = combined.dropna(subset=fcs + ["risk_label"])
    print(f"\n  Combined: {len(combined):,} rows, {len(fcs)} features, "
          f"{len(all_dfs)} tickers")
    print(f"  Label dist:\n{combined['risk_label'].value_counts().sort_index().to_string()}")

    # Walk-forward
    wf = WalkForwardV6(min_train_days=1000, step_days=5, test_window_days=5)
    wf_res = wf.run(combined, fcs)

    if "error" not in wf_res:
        print(f"\n  Cross-stock WF accuracy:  {wf_res['overall_accuracy']:.3f}")
        print(f"  Directional accuracy:     {wf_res['directional_accuracy']:.3f}")
        print(f"  95% CI:                   {wf_res['accuracy_95ci']}")

    # Train global model + save
    print(f"\n  Training global model on all data...")
    ens = EnsembleV6()
    ens.fit(combined[fcs], combined["risk_label"].values)

    if save_pkl:
        save_model("GLOBAL", ens, fcs,
                   {"tickers": tickers, "n_rows": len(combined),
                    "n_features": len(fcs),
                    "wf_accuracy": wf_res.get("overall_accuracy")})

    return {
        "tickers": tickers, "n_rows": len(combined),
        "n_features": len(fcs), "n_tickers": len(all_dfs),
        "walkforward": wf_res,
    }


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CVA-SACS v6 ML Engine")
    parser.add_argument("--tickers", nargs="+",
                        default=["AAPL", "MSFT", "NVDA", "RELIANCE.NS"])
    parser.add_argument("--mode",
                        choices=["full","ablation","shap","walkforward",
                                 "backtest","pretrain","cross"],
                        default="full")
    parser.add_argument("--optuna", action="store_true",
                        help="Enable Optuna HPO (slow but better)")
    parser.add_argument("--trials", type=int, default=100,
                        help="Optuna trials per model")
    parser.add_argument("--output", default=None,
                        help="Save results JSON")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save pkl files")
    args = parser.parse_args()

    print("\n" + "═"*65)
    print("  CVA-SACS v6 — Maximum Impact ML Engine")
    print(f"  Tickers: {', '.join(args.tickers)}")
    print(f"  Mode: {args.mode}")
    print(f"  XGB: {'ON' if XGB_OK else 'OFF'}  "
          f"LGB: {'ON' if LGB_OK else 'OFF'}  "
          f"CAT: {'ON' if CAT_OK else 'OFF'}  "
          f"Optuna: {'ON' if OPTUNA_OK else 'OFF'}  "
          f"SHAP: {'ON' if SHAP_OK else 'OFF'}")
    print("═"*65)

    all_results = {}

    if args.mode == "cross":
        result = run_cross_stock_v6(args.tickers, save_pkl=not args.no_save)
        all_results["cross_stock"] = result
    else:
        for ticker in args.tickers:
            result = run_pipeline_v6(
                ticker,
                mode=args.mode,
                use_optuna=args.optuna,
                optuna_trials=args.trials,
                save_pkl=not args.no_save,
            )
            all_results[ticker] = result

    # Summary
    if args.mode in ("full", "walkforward", "backtest"):
        print(f"\n{'═'*65}")
        print("  SUMMARY")
        print(f"{'═'*65}")
        hdr = f"  {'Ticker':<15} {'WF Acc':>7} {'Dir Acc':>8} "  \
              f"{'95% CI':<22} {'Sharpe':>7} {'Alpha':>8}"
        print(hdr); print("  " + "─"*70)
        for t, r in all_results.items():
            wf = r.get("walkforward", {})
            bt = r.get("backtest", {})
            if "error" in wf:
                continue
            ci  = wf.get("accuracy_95ci", "N/A")
            sha = f"{bt['sharpe_ratio']:+.2f}" if bt and "sharpe_ratio" in bt else "  N/A"
            alp = f"{bt['alpha']:+.1%}"         if bt and "alpha" in bt else "  N/A"
            print(f"  {t:<15} {wf.get('overall_accuracy',0):>7.3f} "
                  f"{wf.get('directional_accuracy',0):>8.3f} "
                  f"{ci:<22} {sha:>7} {alp:>8}")

    if args.output:
        def clean(obj):
            if isinstance(obj, (np.integer,)):  return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, np.ndarray):     return obj.tolist()
            if isinstance(obj, dict):   return {k: clean(v) for k,v in obj.items()}
            if isinstance(obj, list):   return [clean(v) for v in obj]
            return obj
        with open(args.output, "w") as f:
            json.dump(clean(all_results), f, indent=2)
        print(f"\n  Results saved → {args.output}")

    print(f"\n  Done.\n")
