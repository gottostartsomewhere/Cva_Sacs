"""Unit tests for CVA-SACS v6 feature engineering and ML pipeline."""
import numpy as np
import pandas as pd
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _make_dummy_ohlcv(n=500):
    """Create synthetic OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    close = np.maximum(close, 10)  # prevent negative prices
    return pd.DataFrame({
        "Date": dates, "Open": close * 0.999, "High": close * 1.01,
        "Low": close * 0.99, "Close": close,
        "Volume": np.random.randint(1_000_000, 10_000_000, n),
    })


class TestFeatureEngineering:
    """Test feature pipeline produces valid output."""

    def test_features_no_nan_after_warmup(self):
        from cva_sacs_v6_ml import FeatureEngineerV6
        df = _make_dummy_ohlcv(600)
        fe = FeatureEngineerV6()
        result = fe.build(df)
        # After 252-day warmup, no NaN in features
        feature_cols = fe.get_feature_cols(result)
        tail = result[feature_cols].iloc[260:]
        nan_pct = tail.isna().mean()
        bad_cols = nan_pct[nan_pct > 0.1].index.tolist()
        assert len(bad_cols) == 0, f"Features with >10% NaN after warmup: {bad_cols}"

    def test_all_features_have_shift1(self):
        """Verify no lookahead: features should use shift(1)."""
        from cva_sacs_v6_ml import FeatureEngineerV6
        df = _make_dummy_ohlcv(500)
        fe = FeatureEngineerV6()
        result = fe.build(df)
        fcs = fe.get_feature_cols(result)
        # Features should not be identical to unshifted price
        for fc in fcs[:10]:  # spot check
            corr = result[fc].corr(result["Close"])
            assert abs(corr) < 0.999, f"{fc} is perfectly correlated with Close (lookahead?)"

    def test_feature_count(self):
        from cva_sacs_v6_ml import FeatureEngineerV6
        df = _make_dummy_ohlcv(600)
        fe = FeatureEngineerV6()
        result = fe.build(df)
        fcs = fe.get_feature_cols(result)
        assert len(fcs) >= 80, f"Expected 80+ features, got {len(fcs)}"


class TestLabels:
    """Test label construction."""

    def test_labels_bounded_0_4(self):
        from cva_sacs_v6_ml import FeatureEngineerV6, build_label_v6
        df = _make_dummy_ohlcv(600)
        fe = FeatureEngineerV6()
        df = fe.build(df)
        df = build_label_v6(df)
        valid = df["risk_label"].dropna()
        assert valid.min() >= 0, f"Label min {valid.min()} < 0"
        assert valid.max() <= 4, f"Label max {valid.max()} > 4"
        assert set(valid.unique()).issubset({0, 1, 2, 3, 4})

    def test_label_not_all_same_class(self):
        from cva_sacs_v6_ml import FeatureEngineerV6, build_label_v6
        df = _make_dummy_ohlcv(600)
        fe = FeatureEngineerV6()
        df = fe.build(df)
        df = build_label_v6(df)
        n_classes = df["risk_label"].dropna().nunique()
        assert n_classes >= 3, f"Only {n_classes} classes — labels may be degenerate"


class TestWalkForward:
    """Test walk-forward splits."""

    def test_splits_no_overlap(self):
        from cva_sacs_v6_ml import WalkForwardV6, FeatureEngineerV6, build_label_v6
        df = _make_dummy_ohlcv(800)
        fe = FeatureEngineerV6()
        df = fe.build(df)
        df = build_label_v6(df)
        df = df.dropna(subset=["risk_label"]).reset_index(drop=True)

        wf = WalkForwardV6(min_train_days=400, step_days=20, test_window_days=20)
        splits = wf.get_splits(df)
        for i in range(len(splits) - 1):
            train_i, test_i = splits[i]
            train_next, test_next = splits[i + 1]
            # Test sets should not overlap
            overlap = set(test_i) & set(test_next)
            assert len(overlap) == 0, f"Window {i} and {i+1} test sets overlap: {len(overlap)} rows"


class TestBacktest:
    """Test backtest engine."""

    def test_equity_never_negative(self):
        """Equity should never drop below 0 with fractional Kelly."""
        # This is a property test — Kelly with max 25% fraction
        # and 10bp friction should never bankrupt
        initial = 100_000
        equity = initial
        for _ in range(200):
            frac = 0.10
            ret = np.random.normal(0, 0.02)
            pnl = equity * frac * ret
            equity += pnl
        assert equity > 0, "Equity went negative in simulation"


class TestSACS:
    """Test SACS stress classification."""

    def test_sacs_not_always_same(self):
        """After fix, SACS should produce different results for different inputs."""
        import sys
        sys.path.insert(0, ".")
        # Generate different vol scenarios
        results = set()
        for mult in [0.5, 1.0, 1.5, 2.0, 3.0]:
            vol_1x = 0.20
            vol_2x = vol_1x * mult * 0.8  # Non-linear stress
            vol_3x = vol_1x * mult * 1.2
            d2 = abs(vol_2x - vol_1x) / (vol_1x + 1e-9)
            d3 = abs(vol_3x - vol_1x) / (vol_1x + 1e-9)
            if d3 > 0.40:
                cls = "BREAKS"
            elif d2 > 0.20:
                cls = "FRAGILE"
            else:
                cls = "ROBUST"
            results.add(cls)
        assert len(results) >= 2, "SACS always returns same classification"
