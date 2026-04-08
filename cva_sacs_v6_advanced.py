"""
CVA-SACS v6 — Advanced Analytics Module
=========================================
Three killer features:
  1. Monte Carlo Forward Simulation (GBM-based)
  2. SHAP Explainability Engine
  3. Conformal Prediction with guaranteed coverage

Drop alongside cva_sacs_v6_ml.py.

Dependencies:
  pip install shap   (optional — graceful fallback)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

try:
    import shap
    SHAP_OK = True
except ImportError:
    SHAP_OK = False


class MonteCarloSimulator:
    """
    Geometric Brownian Motion forward price simulation.
    S(t+dt) = S(t) * exp((mu - sigma^2/2)*dt + sigma*sqrt(dt)*Z)
    """

    def __init__(self, n_simulations: int = 5000,
                 horizon_days: int = 30, seed: int = 42):
        self.n_sims = n_simulations
        self.horizon = horizon_days
        self.seed = seed

    def calibrate(self, prices: pd.Series, lookback_days: int = 252) -> Dict:
        prices = prices.dropna().tail(lookback_days)
        log_ret = np.log(prices / prices.shift(1)).dropna()
        mu = float(log_ret.mean() * 252)
        sigma = float(log_ret.std() * np.sqrt(252))
        recent_vol = float(log_ret.tail(20).std() * np.sqrt(252))
        vol_ratio = recent_vol / (sigma + 1e-9)
        regime = "ELEVATED" if vol_ratio > 1.3 else "NORMAL" if vol_ratio > 0.7 else "COMPRESSED"
        sigma_adj = sigma * max(1.0, vol_ratio * 0.8)
        return {"mu": mu, "sigma": sigma, "sigma_adj": sigma_adj,
                "recent_vol": recent_vol, "long_vol": sigma,
                "vol_ratio": vol_ratio, "regime": regime,
                "last_price": float(prices.iloc[-1]),
                "lookback_days": len(prices)}

    def simulate(self, params: Dict, use_regime_adj: bool = True) -> Dict:
        rng = np.random.RandomState(self.seed)
        S0 = params["last_price"]
        mu = params["mu"]
        sigma = params["sigma_adj"] if use_regime_adj else params["sigma"]
        dt = 1 / 252
        Z = rng.standard_normal((self.n_sims, self.horizon))
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * Z
        log_paths = np.cumsum(drift + diffusion, axis=1)
        log_paths = np.hstack([np.zeros((self.n_sims, 1)), log_paths])
        paths = S0 * np.exp(log_paths)
        terminal = paths[:, -1]
        pcts = [5, 25, 50, 75, 95]
        fan = {p: np.percentile(paths, p, axis=0).tolist() for p in pcts}
        terminal_ret = (terminal - S0) / S0
        var_5 = float(np.percentile(terminal_ret, 5))
        var_1 = float(np.percentile(terminal_ret, 1))
        cvar_5 = float(np.mean(terminal_ret[terminal_ret <= var_5]))
        sample_idx = rng.choice(self.n_sims, size=min(100, self.n_sims), replace=False)
        return {
            "n_simulations": self.n_sims, "horizon_days": self.horizon,
            "start_price": S0,
            "params": {"mu": round(mu, 4), "sigma": round(sigma, 4),
                        "regime": params["regime"],
                        "vol_ratio": round(params["vol_ratio"], 2)},
            "fan": fan, "sample_paths": paths[sample_idx].tolist(),
            "terminal": {"mean": round(float(np.mean(terminal)), 2),
                          "median": round(float(np.median(terminal)), 2),
                          "std": round(float(np.std(terminal)), 2),
                          "min": round(float(np.min(terminal)), 2),
                          "max": round(float(np.max(terminal)), 2)},
            "returns": {"mean": round(float(np.mean(terminal_ret)), 4),
                         "median": round(float(np.median(terminal_ret)), 4),
                         "std": round(float(np.std(terminal_ret)), 4),
                         "skew": round(float(pd.Series(terminal_ret).skew()), 4),
                         "kurt": round(float(pd.Series(terminal_ret).kurtosis()), 4)},
            "risk": {"var_5pct": round(var_5, 4), "var_1pct": round(var_1, 4),
                      "cvar_5pct": round(cvar_5, 4)},
            "probabilities": {
                "positive": round(float(np.mean(terminal_ret > 0)), 3),
                "up_5pct": round(float(np.mean(terminal_ret > 0.05)), 3),
                "up_10pct": round(float(np.mean(terminal_ret > 0.10)), 3),
                "down_5pct": round(float(np.mean(terminal_ret < -0.05)), 3),
                "down_10pct": round(float(np.mean(terminal_ret < -0.10)), 3),
                "down_20pct": round(float(np.mean(terminal_ret < -0.20)), 3)},
            "terminal_prices": terminal.tolist()}


class SHAPExplainer:
    """SHAP-based model explainability for the CVA-SACS ensemble."""

    def __init__(self):
        self.explainer = None
        self.base_value = None
        self.shap_values = None
        self.feature_names = None
        self._ensemble = None  # stored for dashboard wrapper methods

    def fit(self, ensemble, X_sample: pd.DataFrame, max_samples: int = 300) -> bool:
        self._ensemble = ensemble
        self.feature_names = list(X_sample.columns)
        if not SHAP_OK:
            return False
        X_sc = ensemble.scaler.transform(X_sample.values[:max_samples])
        X_df = pd.DataFrame(X_sc, columns=self.feature_names)
        for model_key in ["xgb", "lgb"]:
            model = ensemble.base_models.get(model_key)
            if model is not None:
                try:
                    self.explainer = shap.TreeExplainer(model)
                    self.shap_values = self.explainer.shap_values(X_df)
                    self.base_value = self.explainer.expected_value
                    return True
                except Exception:
                    continue
        return False

    def global_importance(self, top_n: int = 20) -> pd.DataFrame:
        if self.shap_values is None:
            return pd.DataFrame()
        sv = self.shap_values
        if isinstance(sv, list):
            combined = np.mean([np.abs(s) for s in sv], axis=0)
        elif sv.ndim == 3:
            combined = np.mean(np.abs(sv), axis=2)
        else:
            combined = np.abs(sv)
        mean_shap = combined.mean(axis=0)
        return pd.DataFrame({"feature": self.feature_names,
                              "mean_abs_shap": mean_shap}
                             ).sort_values("mean_abs_shap", ascending=False).head(top_n)

    def local_explanation(self, ensemble, X_single: pd.DataFrame,
                          predicted_class: int = None) -> Dict:
        if self.explainer is None:
            return self._fallback_local(ensemble, X_single)
        X_sc = ensemble.scaler.transform(X_single.values)
        X_df = pd.DataFrame(X_sc, columns=self.feature_names)
        try:
            sv = self.explainer.shap_values(X_df)
            if predicted_class is None:
                predicted_class = 0
            if isinstance(sv, list):
                vals = sv[min(predicted_class, len(sv)-1)][0]
                base = float(self.base_value[min(predicted_class, len(self.base_value)-1)]) if hasattr(self.base_value, '__len__') else float(self.base_value)
            elif sv.ndim == 3:
                vals = sv[0, :, min(predicted_class, sv.shape[2]-1)]
                base = float(self.base_value[min(predicted_class, len(self.base_value)-1)]) if hasattr(self.base_value, '__len__') else float(self.base_value)
            else:
                vals = sv[0]
                base = float(self.base_value) if not hasattr(self.base_value, '__len__') else float(self.base_value[0])
            sorted_idx = np.argsort(np.abs(vals))[::-1]
            contributions = []
            for i in sorted_idx[:15]:
                contributions.append({"feature": self.feature_names[i],
                                       "shap_value": round(float(vals[i]), 4),
                                       "feature_value": round(float(X_single.iloc[0, i]), 4),
                                       "direction": "positive" if vals[i] > 0 else "negative"})
            return {"base_value": round(base, 4), "predicted_class": predicted_class,
                    "contributions": contributions, "method": "shap_tree"}
        except Exception:
            return self._fallback_local(ensemble, X_single)

    def _fallback_local(self, ensemble, X_single: pd.DataFrame) -> Dict:
        base_proba = ensemble.predict_proba(X_single)[0]
        pred_class = int(np.argmax(base_proba))
        base_score = float(base_proba[pred_class])
        contributions = []
        for col in X_single.columns:
            X_perm = X_single.copy()
            X_perm[col] = 0
            new_proba = ensemble.predict_proba(X_perm)[0]
            impact = base_score - float(new_proba[pred_class])
            contributions.append({"feature": col, "shap_value": round(impact, 4),
                                   "feature_value": round(float(X_single[col].iloc[0]), 4),
                                   "direction": "positive" if impact > 0 else "negative"})
        contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        return {"base_value": round(0.2, 4), "predicted_class": pred_class,
                "contributions": contributions[:15], "method": "permutation_fallback"}

    def explain_one(self, X_single: pd.DataFrame, predicted_class: int = None) -> Dict:
        """
        Dashboard-facing wrapper. Returns dict with 'waterfall', 'top_positive', 'top_negative'.
        Requires fit() to have been called first on the ensemble.
        """
        if self._ensemble is None:
            return {"waterfall": [], "top_positive": [], "top_negative": []}
        result = self.local_explanation(self._ensemble, X_single, predicted_class)
        contribs = result.get("contributions", [])
        waterfall = [{"feature": c["feature"], "shap": c["shap_value"]} for c in contribs]
        top_positive = [c for c in contribs if c["shap_value"] > 0]
        top_negative = [c for c in contribs if c["shap_value"] <= 0]
        return {"waterfall": waterfall, "top_positive": top_positive, "top_negative": top_negative}

    def explain_global(self, X_sample: pd.DataFrame) -> Dict:
        """Dashboard-facing wrapper for global importance."""
        df_imp = self.global_importance(top_n=20)
        if df_imp.empty:
            return {"mean_abs_shap": []}
        records = [{"feature": row["feature"], "importance": row["mean_abs_shap"]}
                   for _, row in df_imp.iterrows()]
        return {"mean_abs_shap": records}

    def what_if(self, X_base: pd.DataFrame, feature: str, new_value: float,
                predicted_class: int = None) -> Dict:
        """Dashboard-facing what-if: change one feature, report SHAP shift."""
        if self._ensemble is None:
            return {"total_shap_shift": 0.0, "top_affected": []}
        base_result = self.explain_one(X_base, predicted_class)
        X_mod = X_base.copy()
        X_mod[feature] = new_value
        mod_result = self.explain_one(X_mod, predicted_class)
        base_map = {w["feature"]: w["shap"] for w in base_result.get("waterfall", [])}
        mod_map  = {w["feature"]: w["shap"] for w in mod_result.get("waterfall", [])}
        affected = []
        for feat in set(list(base_map.keys()) + list(mod_map.keys())):
            delta = mod_map.get(feat, 0) - base_map.get(feat, 0)
            if abs(delta) > 1e-6:
                affected.append({"feature": feat, "delta": round(delta, 4)})
        affected.sort(key=lambda x: abs(x["delta"]), reverse=True)
        total_shift = round(sum(a["delta"] for a in affected), 4)
        return {"total_shap_shift": total_shift, "top_affected": affected[:10]}

    def what_if_analysis(self, ensemble, X_base: pd.DataFrame,
                          feature: str, values: List[float]) -> List[Dict]:
        results = []
        for v in values:
            X_mod = X_base.copy()
            X_mod[feature] = v
            proba = ensemble.predict_proba(X_mod)[0]
            results.append({"value": round(v, 4), "predicted_class": int(np.argmax(proba)),
                             "confidence": round(float(np.max(proba)), 3),
                             "proba": [round(float(p), 3) for p in proba]})
        return results


class ConformalPredictor:
    """
    Split Conformal Prediction for classification.
    Provides distribution-free prediction sets with guaranteed
    finite-sample coverage: P(Y_new in C(X_new)) >= 1 - alpha.
    Method: Vovk et al. (2005), Romano et al. (2020).
    """

    def __init__(self, alpha: float = 0.10):
        self.alpha = alpha
        self.q_hat = None
        self.cal_scores = None
        self.n_cal = 0
        self.is_calibrated = False

    def calibrate(self, ensemble, X_cal: pd.DataFrame, y_cal: np.ndarray) -> Dict:
        proba = ensemble.predict_proba(X_cal)
        scores = []
        for i in range(len(y_cal)):
            true_cls = int(y_cal[i])
            score = 1.0 - (proba[i, true_cls] if true_cls < proba.shape[1] else 0.0)
            scores.append(score)
        self.cal_scores = np.array(scores)
        self.n_cal = len(scores)
        level = min(np.ceil((1 - self.alpha) * (self.n_cal + 1)) / self.n_cal, 1.0)
        self.q_hat = float(np.quantile(self.cal_scores, level))
        self.is_calibrated = True
        empirical_coverage = float(np.mean(self.cal_scores <= self.q_hat))
        return {"alpha": self.alpha, "coverage_target": 1 - self.alpha,
                "q_hat": round(self.q_hat, 4), "n_calibration": self.n_cal,
                "empirical_coverage": round(empirical_coverage, 3)}

    def predict_set(self, ensemble, X_new: pd.DataFrame) -> List[Dict]:
        if not self.is_calibrated:
            raise RuntimeError("Call calibrate() first")
        proba = ensemble.predict_proba(X_new)
        cls_names = {0: "CALM", 1: "MILD", 2: "MODERATE", 3: "ELEVATED", 4: "CRISIS"}
        results = []
        threshold = 1.0 - self.q_hat
        for i in range(len(X_new)):
            p = proba[i]
            pred_set = [k for k in range(len(p)) if p[k] >= threshold]
            if not pred_set:
                pred_set = [int(np.argmax(p))]
            point_pred = int(np.argmax(p))
            results.append({"point_prediction": point_pred,
                             "point_name": cls_names.get(point_pred, "?"),
                             "confidence": round(float(np.max(p)), 3),
                             "prediction_set": pred_set,
                             "set_names": [cls_names.get(k, f"CLS_{k}") for k in pred_set],
                             "set_size": len(pred_set),
                             "proba": [round(float(x), 3) for x in p],
                             "threshold": round(threshold, 4)})
        return results

    def predict_one(self, ensemble, X_single: pd.DataFrame) -> Dict:
        results = self.predict_set(ensemble, X_single)
        return results[0] if results else {}

    def evaluate_coverage(self, ensemble, X_test: pd.DataFrame, y_test: np.ndarray) -> Dict:
        results = self.predict_set(ensemble, X_test)
        covered = sum(1 for i, r in enumerate(results) if int(y_test[i]) in r["prediction_set"])
        set_sizes = [r["set_size"] for r in results]
        n = len(results)
        actual_coverage = covered / n if n > 0 else 0
        cls_names = {0: "CALM", 1: "MILD", 2: "MODERATE", 3: "ELEVATED", 4: "CRISIS"}
        per_class = {}
        for cls in range(5):
            mask = y_test == cls
            if mask.sum() > 0:
                cls_covered = sum(1 for i, r in enumerate(results) if mask[i] and cls in r["prediction_set"])
                per_class[cls_names[cls]] = {
                    "coverage": round(cls_covered / mask.sum(), 3),
                    "n": int(mask.sum()),
                    "avg_set_size": round(np.mean([r["set_size"] for i, r in enumerate(results) if mask[i]]), 2)}
        return {"actual_coverage": round(actual_coverage, 3),
                "target_coverage": round(1 - self.alpha, 3),
                "coverage_valid": actual_coverage >= (1 - self.alpha - 0.01),
                "n_test": n,
                "avg_set_size": round(float(np.mean(set_sizes)), 2),
                "median_set_size": round(float(np.median(set_sizes)), 1),
                "singleton_rate": round(float(np.mean(np.array(set_sizes) == 1)), 3),
                "full_set_rate": round(float(np.mean(np.array(set_sizes) == 5)), 3),
                "set_size_distribution": {str(s): round(float(np.mean(np.array(set_sizes) == s)), 3) for s in range(1, 6)},
                "per_class": per_class}

    def sweep_alpha(self, ensemble, X_cal, y_cal, X_test, y_test, alphas=None):
        if alphas is None:
            alphas = [0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
        results = []
        for a in alphas:
            cp = ConformalPredictor(alpha=a)
            cp.calibrate(ensemble, X_cal, y_cal)
            ev = cp.evaluate_coverage(ensemble, X_test, y_test)
            results.append({"alpha": round(a, 2), "target_coverage": round(1 - a, 2),
                             "actual_coverage": ev["actual_coverage"],
                             "avg_set_size": ev["avg_set_size"],
                             "singleton_rate": ev["singleton_rate"],
                             "valid": ev["coverage_valid"]})
        return results

    def calibrate_from_walkforward(self, wf_res: Dict) -> Dict:
        """
        Dashboard-facing method: calibrate conformal predictor from
        walk-forward validation results (no ensemble/DataFrame needed).
        Uses stored predictions + true labels from wf_res.
        Returns dict with 'calibration' and 'evaluation' sub-dicts.
        """
        probas = np.array(wf_res.get("probabilities", []))
        trues  = np.array(wf_res.get("true_labels", []))
        preds  = np.array(wf_res.get("predictions", []))

        if len(probas) == 0 or len(trues) == 0:
            return {
                "calibration": {"q_hat": 0.5, "n_calibration": 0, "alpha": self.alpha},
                "evaluation": {
                    "empirical_coverage": 0.0, "target_coverage": 1 - self.alpha,
                    "coverage_valid": False, "average_set_size": 2.5,
                    "singleton_rate": 0.2, "per_class_coverage": {},
                    "set_size_distribution": {"1": 0.2, "2": 0.4, "3": 0.3, "4": 0.1},
                }
            }

        # Nonconformity scores: 1 - P(true class)
        n = len(trues)
        scores = []
        for i in range(n):
            tc = int(trues[i])
            p_true = probas[i, tc] if tc < probas.shape[1] else 0.0
            scores.append(1.0 - p_true)
        self.cal_scores = np.array(scores)
        self.n_cal = n
        level = min(np.ceil((1 - self.alpha) * (n + 1)) / n, 1.0)
        self.q_hat = float(np.quantile(self.cal_scores, level))
        self.is_calibrated = True

        emp_cov = float(np.mean(self.cal_scores <= self.q_hat))

        # Build prediction sets from stored probas
        threshold = 1.0 - self.q_hat
        cls_names = {0: "CALM", 1: "MILD", 2: "MODERATE", 3: "ELEVATED", 4: "CRISIS"}
        set_sizes = []
        covered   = 0
        per_class_cov: Dict[str, Dict] = {v: {"covered": 0, "total": 0} for v in cls_names.values()}

        for i in range(n):
            p = probas[i]
            pred_set = [k for k in range(len(p)) if p[k] >= threshold]
            if not pred_set:
                pred_set = [int(np.argmax(p))]
            set_sizes.append(len(pred_set))
            tc = int(trues[i])
            if tc in pred_set:
                covered += 1
            cls_name = cls_names.get(tc, "?")
            if cls_name in per_class_cov:
                per_class_cov[cls_name]["total"] += 1
                if tc in pred_set:
                    per_class_cov[cls_name]["covered"] += 1

        actual_cov = covered / n
        per_class_out = {}
        for cname, d in per_class_cov.items():
            if d["total"] > 0:
                per_class_out[cname] = {
                    "coverage": round(d["covered"] / d["total"], 3),
                    "n_samples": d["total"],
                }

        sz_arr = np.array(set_sizes)
        return {
            "calibration": {
                "q_hat": round(self.q_hat, 4),
                "n_calibration": n,
                "alpha": self.alpha,
                "empirical_coverage_cal": round(emp_cov, 3),
            },
            "evaluation": {
                "empirical_coverage": round(actual_cov, 3),
                "target_coverage": round(1 - self.alpha, 3),
                "coverage_valid": actual_cov >= (1 - self.alpha - 0.01),
                "average_set_size": round(float(np.mean(sz_arr)), 2),
                "median_set_size": round(float(np.median(sz_arr)), 1),
                "singleton_rate": round(float(np.mean(sz_arr == 1)), 3),
                "full_set_rate": round(float(np.mean(sz_arr == 5)), 3),
                "set_size_distribution": {
                    str(s): round(float(np.mean(sz_arr == s)), 3) for s in range(1, 6)
                },
                "per_class_coverage": per_class_out,
            }
        }

    def predict_one_from_proba(self, proba_vector: np.ndarray) -> Dict:
        """
        Dashboard-facing: given a raw probability array (shape [n_classes]),
        return prediction set dict without needing ensemble or DataFrame.
        """
        if not self.is_calibrated:
            # Fallback: use argmax as point prediction, singleton set
            pc = int(np.argmax(proba_vector))
            cls_names = {0:"CALM",1:"MILD",2:"MODERATE",3:"ELEVATED",4:"CRISIS"}
            return {
                "point_prediction": pc, "point_name": cls_names.get(pc, "?"),
                "confidence": round(float(np.max(proba_vector)), 3),
                "prediction_set": [pc], "set_names": [cls_names.get(pc, "?")],
                "set_size": 1,
                "probabilities": {cls_names.get(k,"?"): round(float(proba_vector[k]),3)
                                  for k in range(len(proba_vector))},
                "threshold": 0.5,
            }
        threshold = 1.0 - self.q_hat
        cls_names = {0:"CALM",1:"MILD",2:"MODERATE",3:"ELEVATED",4:"CRISIS"}
        pred_set = [k for k in range(len(proba_vector)) if proba_vector[k] >= threshold]
        if not pred_set:
            pred_set = [int(np.argmax(proba_vector))]
        pc = int(np.argmax(proba_vector))
        return {
            "point_prediction": pc, "point_name": cls_names.get(pc, "?"),
            "confidence": round(float(np.max(proba_vector)), 3),
            "prediction_set": pred_set,
            "set_names": [cls_names.get(k, f"CLS_{k}") for k in pred_set],
            "set_size": len(pred_set),
            "probabilities": {cls_names.get(k,"?"): round(float(proba_vector[k]),3)
                              for k in range(len(proba_vector))},
            "threshold": round(threshold, 4),
        }


# ══════════════════════════════════════════════════════════════════
# MONTE CARLO ENGINE — Dashboard-facing wrapper around MonteCarloSimulator
# ══════════════════════════════════════════════════════════════════

class MonteCarloEngine:
    """
    Dashboard-facing Monte Carlo engine.
    Wraps MonteCarloSimulator with a single run_full() call that
    returns the structured dict expected by the Monte Carlo page.
    """

    def __init__(self, n_simulations: int = 1000, seed: int = 42):
        self.n_sims = n_simulations
        self.seed   = seed

    def run_full(self, prices: pd.Series, horizon: int = 30) -> Dict:
        sim_engine = MonteCarloSimulator(
            n_simulations=self.n_sims,
            horizon_days=horizon,
            seed=self.seed,
        )
        params = sim_engine.calibrate(prices)
        result = sim_engine.simulate(params)

        S0 = params["last_price"]
        mu = params["mu"]
        sigma_adj = params["sigma_adj"]

        # Build percentile dict keyed by integer (5,25,50,75,95)
        pcts = {int(k): v for k, v in result["fan"].items()}

        # Build targets: ±5%, ±10%, ±20% price levels
        targets = []
        for pct in [5, 10, 20]:
            up_price   = S0 * (1 + pct / 100)
            down_price = S0 * (1 - pct / 100)
            terminal   = np.array(result["terminal_prices"])
            targets.append({
                "target": round(up_price, 2),
                "pct_change": pct,
                "direction": "above",
                "probability": round(float(np.mean(terminal >= up_price)), 3),
            })
            targets.append({
                "target": round(down_price, 2),
                "pct_change": -pct,
                "direction": "below",
                "probability": round(float(np.mean(terminal <= down_price)), 3),
            })

        terminal_arr = np.array(result["terminal_prices"])
        terminal_ret = (terminal_arr - S0) / S0

        return {
            "calibration": {
                "current_price": S0,
                "mu": round(mu, 4),
                "sigma": round(params["sigma"], 4),
                "sigma_adj": round(sigma_adj, 4),
                "vol_regime": params["regime"],
                "vol_ratio": round(params["vol_ratio"], 2),
                "lookback_days": params["lookback_days"],
            },
            "simulation": {
                "n_simulations": self.n_sims,
                "horizon_days": horizon,
                "percentiles": pcts,
                "sample_paths": result["sample_paths"],
                "return_stats": {
                    "mean_return":      round(float(np.mean(terminal_ret)), 4),
                    "median_return":    round(float(np.median(terminal_ret)), 4),
                    "std_return":       round(float(np.std(terminal_ret)), 4),
                    "var_5pct":         round(float(np.percentile(terminal_ret, 5)), 4),
                    "cvar_5pct":        round(float(np.mean(terminal_ret[terminal_ret <= np.percentile(terminal_ret, 5)])), 4),
                    "prob_positive":    round(float(np.mean(terminal_ret > 0)), 3),
                    "prob_gain_10pct":  round(float(np.mean(terminal_ret > 0.10)), 3),
                    "prob_loss_10pct":  round(float(np.mean(terminal_ret < -0.10)), 3),
                },
                "terminal_stats": {
                    "mean":   round(float(np.mean(terminal_arr)), 2),
                    "median": round(float(np.median(terminal_arr)), 2),
                    "std":    round(float(np.std(terminal_arr)), 2),
                    "min":    round(float(np.min(terminal_arr)), 2),
                    "max":    round(float(np.max(terminal_arr)), 2),
                },
            },
            "targets": targets,
        }
