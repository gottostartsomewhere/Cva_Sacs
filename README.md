# CVA-SACS v6 — Market Intelligence Terminal

## Quick Start
```bash
pip install -r requirements.txt
streamlit run cva_sacs_v6.py
```

## Docker
```bash
docker build -t cva-sacs-v6 .
docker run -p 8501:8501 cva-sacs-v6
```

## Architecture
- `cva_sacs_v6.py` — Streamlit dashboard (9 pages)
- `cva_sacs_v6_ml.py` — ML engine (130 features, XGB+LGB+CAT ensemble)
- `cva_sacs_v6_sentiment.py` — FinBERT NLP sentiment
- `cva_sacs_v6_data.py` — Data expansion (S&P 500 + Nifty 100)
- `cva_sacs_v6_advanced.py` — SHAP, Monte Carlo, Conformal Prediction
- `config.py` — Centralised configuration
- `tests/` — Unit tests (pytest)
