<div align="center">

# CVA-SACS

**Cascading Veto Architecture · Stress-Adjusted Confidence Scoring**

A research-grade equity stress-testing terminal — 130 engineered features, a calibrated tri-model ensemble, FinBERT sentiment, walk-forward backtesting, Monte Carlo paths, SHAP explainability, and conformal prediction intervals.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-017CEE?style=flat-square)
![LightGBM](https://img.shields.io/badge/LightGBM-2B7A0B?style=flat-square)
![CatBoost](https://img.shields.io/badge/CatBoost-FFCC00?style=flat-square&logoColor=black)
![FinBERT](https://img.shields.io/badge/FinBERT-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-000?style=flat-square)

</div>

---

## What this is

CVA-SACS is a **market regime classifier and stress terminal** — not a signal generator that promises returns. It ingests price, volatility, cross-asset macro, and news sentiment for ~600 tickers (S&P 500 + Nifty 100), engineers 130 features across 8 analytical layers, then classifies each name into a 5-class risk regime:

| Class     | CRI range | Meaning                                              |
|-----------|-----------|------------------------------------------------------|
| `CALM`    | 0–20      | Stable vol, broad participation, low drawdown risk   |
| `MILD`    | 20–40     | Slight dispersion, early warning signals             |
| `MODERATE`| 40–60     | Elevated vol-of-vol, cross-asset divergence          |
| `ELEVATED`| 60–80     | Breadth deterioration, macro stress                  |
| `CRISIS`  | 80–100    | Regime break — volatility clustering, tail risk     |

The **Compound Risk Index (CRI)** is a 0–100 score distilled from the ensemble's posterior — the headline number the dashboard revolves around.

---

## Why the architecture is named the way it is

- **CVA — Cascading Veto Architecture.** A signal has to survive sequential vetoes (vol regime → cross-asset confirmation → sentiment check → drawdown screen) before it becomes actionable. One failed layer kills the call.
- **SACS — Stress-Adjusted Confidence Scoring.** Model confidence isn't raw probability; it's scaled by the stress signature of the current regime. High-confidence calls in `CALM` and in `CRISIS` mean different things — SACS makes that explicit.

---

## Architecture

```
┌───────────────────────────────────────────────────────────────────────┐
│                       DATA LAYER  (yfinance, FRED)                    │
│   S&P 500 + Nifty 100 · OHLCV · options · FINRA short interest        │
└───────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌───────────────────────────────────────────────────────────────────────┐
│            FEATURE ENGINEERING  —  130 features · 8 layers            │
│  A  OHLCV technicals        E  Cross-asset macro (VIX, DXY, yields)   │
│  B  Rolling vol / skew      F  Alt-data (short interest, flows)       │
│  C  Momentum persistence    G  Signal persistence / half-life         │
│  D  Drawdown geometry       H  FinBERT news sentiment                 │
└───────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌───────────────────────────────────────────────────────────────────────┐
│              ENSEMBLE  —  XGBoost · LightGBM · CatBoost               │
│     Optuna (300 trials) · Logistic meta-learner · Isotonic calibrate  │
└───────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌───────────────────────────────────────────────────────────────────────┐
│       RISK LAYER  —  CRI · 5-class regime · Conformal intervals       │
│              Walk-forward backtest · Kelly-sized sleeve               │
│              Monte Carlo (GBM paths) · SHAP attribution               │
└───────────────────────────────────────────────────────────────────────┘
```

---

## What makes it different from a toy notebook

**Label engineering.** The target isn't "next-day up/down." It's a dual-horizon composite: `0.45 × 5d Sharpe + 0.35 × 10d Sharpe − 0.20 × max drawdown`. The model is penalized for predicting rallies that come with tail events.

**Walk-forward, not k-fold.** All evaluation uses expanding-window walk-forward with purged boundaries — no look-ahead, no leakage from post-split normalization.

**Calibration, not just accuracy.** Raw ensemble outputs are passed through isotonic regression per class, then combined by a logistic meta-learner. The output is a probability you can actually size against.

**Conformal intervals.** Transductive conformal prediction wraps the classifier — instead of "class = ELEVATED," you get "class = ELEVATED with 90% coverage, set {MODERATE, ELEVATED}."

**Explainability built in.** Every prediction exposes SHAP attribution via `TreeExplainer`. The dashboard surfaces the top 10 contributing features per ticker per day.

**Position sizing, not just direction.** A Kelly-fractioned allocator converts calibrated probability + realized vol into position weights, with 10 bp round-trip friction baked into the backtest.

---

## The 8-page terminal

1. **Market Overview** — regime distribution, CRI heatmap, breadth
2. **Ticker Deep Dive** — per-name CRI, SHAP waterfall, conformal set
3. **Ensemble Inspector** — per-model probabilities, disagreement surface
4. **Sentiment** — FinBERT scores from news headlines, rolling z-score
5. **Macro Panel** — VIX term structure, DXY, yield spreads
6. **Walk-Forward Backtest** — equity curve, hit rate, Sharpe, max DD
7. **Monte Carlo** — forward GBM paths calibrated to current regime
8. **Conformal Dashboard** — coverage, efficiency, prediction-set sizes

---

## Quick start

```bash
# clone + install
git clone https://github.com/gottostartsomewhere/Cva_Sacs.git
cd Cva_Sacs
pip install -r requirements.txt

# run the terminal
streamlit run cva_sacs_v6.py
```

Or with Make:

```bash
make setup   # installs deps
make run     # boots Streamlit on :8501
```

Or Docker:

```bash
docker build -t cva-sacs .
docker run -p 8501:8501 cva-sacs
```

Then open http://localhost:8501.

---

## Project layout

```
Cva_Sacs/
├── cva_sacs_v6.py              # Streamlit entry — 8-page terminal
├── cva_sacs_v6_ml.py           # Ensemble · feature store · calibration
├── cva_sacs_v6_sentiment.py    # FinBERT pipeline · news ingestion
├── cva_sacs_v6_data.py         # Ticker universe · OHLCV loaders
├── cva_sacs_v6_advanced.py     # SHAP · Monte Carlo · conformal
├── config.py                   # Centralised config (paths, tickers, thresholds)
├── tests/
│   └── test_features.py        # Feature-engineering unit tests
├── Dockerfile
├── Makefile
└── requirements.txt
```

---

## Stack

| Layer            | Tools                                                         |
|------------------|---------------------------------------------------------------|
| UI / terminal    | Streamlit, Plotly                                             |
| ML ensemble      | XGBoost, LightGBM, CatBoost, Optuna, scikit-learn (isotonic)  |
| NLP              | HuggingFace Transformers, ProsusAI/FinBERT, PyTorch           |
| Time series      | Prophet, statsmodels                                          |
| Explainability   | SHAP (`TreeExplainer`)                                        |
| Data             | yfinance, FRED, FINRA short-interest feeds                    |
| Infra            | Docker, Make                                                  |

---

## Honest scope

This is a **research terminal**, not a trading system. It classifies regimes and produces calibrated probabilities with intervals — it does **not** place orders, connect to a broker, or handle real money. The walk-forward backtest is simulated on historical data with 10 bp friction; live performance will differ.

The published ~79% directional accuracy is the walk-forward result on the held-out tail of the sample. Treat it as a ceiling, not a promise.

---

## License

MIT © John Kevin
