"""
CVA-SACS v6 — Centralised Configuration
=========================================
All tunable parameters in one place.
"""

# ── Model Hyperparameters ─────────────────────────────────────
XGB_PARAMS = dict(
    n_estimators=1000, max_depth=5, learning_rate=0.04,
    subsample=0.8, colsample_bytree=0.7, gamma=0.1,
    reg_alpha=0.2, reg_lambda=1.5, min_child_weight=10,
)
LGB_PARAMS = dict(
    n_estimators=1000, max_depth=5, learning_rate=0.04,
    subsample=0.8, colsample_bytree=0.7, min_child_samples=15,
    reg_alpha=0.2, reg_lambda=1.5,
)
CAT_PARAMS = dict(
    iterations=800, depth=6, learning_rate=0.04,
    l2_leaf_reg=3.0,
)

# ── Label Construction ────────────────────────────────────────
LABEL_H1 = 5      # short horizon (days)
LABEL_H2 = 10     # long horizon (days)
LABEL_W1 = 0.45   # weight for h1
LABEL_W2 = 0.35   # weight for h2
LABEL_WDD = 0.20  # weight for drawdown penalty

# ── Walk-Forward ──────────────────────────────────────────────
WF_MIN_TRAIN = 504   # minimum training days (~2 years)
WF_STEP = 5          # step size (days)
WF_TEST_WINDOW = 5   # test window (days)

# ── Backtest ──────────────────────────────────────────────────
BT_FRICTION = 0.001          # base friction per side
BT_INITIAL_CAPITAL = 100_000
BT_MAX_KELLY = 0.25

# ── CRI Weights ───────────────────────────────────────────────
CRI_WEIGHTS = {"ML": 0.30, "VaR": 0.25, "SACS": 0.20, "Mom": 0.15, "Vol": 0.10}
CRI_ZONES = {"SAFE": 26, "CAUTION": 51, "ELEVATED": 76}

# ── SACS Thresholds ───────────────────────────────────────────
SACS_FRAGILE_THRESHOLD = 0.20
SACS_BREAKS_THRESHOLD = 0.40

# ── Sentiment ─────────────────────────────────────────────────
SENTIMENT_HALF_LIFE = 3.0    # days
SENTIMENT_BUY_THRESHOLD = 0.10
SENTIMENT_SELL_THRESHOLD = -0.10

# ── Watchlists ────────────────────────────────────────────────
WL_US = {
    "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "NVIDIA",
    "GOOGL": "Alphabet", "AMZN": "Amazon", "META": "Meta",
    "TSLA": "Tesla", "JPM": "JPMorgan", "V": "Visa", "JNJ": "J&J",
}
WL_IN = {
    "RELIANCE.NS": "Reliance", "TCS.NS": "TCS",
    "HDFCBANK.NS": "HDFC Bank", "INFY.NS": "Infosys",
    "ICICIBANK.NS": "ICICI Bank", "HINDUNILVR.NS": "HUL",
    "SBIN.NS": "SBI", "TATACONSUM.NS": "Tata Consumer",
    "WIPRO.NS": "Wipro", "AXISBANK.NS": "Axis Bank",
}

# ── Global Seed ───────────────────────────────────────────────
GLOBAL_SEED = 42

# ── Data ──────────────────────────────────────────────────────
DATA_YEARS = 6
MACRO_TICKERS = {"VIX": "^VIX", "SPY": "SPY", "TLT": "TLT",
                 "HYG": "HYG", "DXY": "DX-Y.NYB", "GLD": "GLD"}
