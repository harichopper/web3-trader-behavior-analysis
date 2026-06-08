"""
analysis.py
-----------
Statistical analysis: EDA, sentiment vs performance tests,
clustering, and machine learning pipeline.

Author : Senior Quant Research Team
Version: 1.0.0
"""

from __future__ import annotations

import warnings
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

from src.utils import PROCESSED_DIR, SENTIMENT_ORDER, RANDOM_SEED, get_logger

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

warnings.filterwarnings("ignore")
logger = get_logger(__name__)


# ─────────────────────────────────────────────
# EDA SUMMARIES
# ─────────────────────────────────────────────

def eda_sentiment(fg: pd.DataFrame) -> pd.DataFrame:
    """Frequency table for sentiment classifications."""
    tbl = fg["classification"].value_counts().rename_axis("sentiment").reset_index(name="count")
    tbl["pct"] = tbl["count"] / tbl["count"].sum() * 100
    logger.info("Sentiment distribution:\n%s", tbl.to_string(index=False))
    return tbl


def eda_trader(df: pd.DataFrame) -> dict[str, Any]:
    """Return a dict of EDA statistics for trader data."""
    result: dict[str, Any] = {}
    result["total_trades"] = len(df)
    result["unique_accounts"] = df["account"].nunique() if "account" in df.columns else 0
    result["unique_symbols"] = df["symbol"].nunique() if "symbol" in df.columns else 0

    pnl = df["closedpnl"]
    result["total_pnl"] = pnl.sum()
    result["avg_pnl"] = pnl.mean()
    result["median_pnl"] = pnl.median()
    result["max_profit"] = pnl.max()
    result["max_loss"] = pnl.min()
    result["win_rate"] = (pnl > 0).mean()
    result["pnl_std"] = pnl.std()

    if "leverage" in df.columns:
        result["avg_leverage"] = df["leverage"].mean()
        result["median_leverage"] = df["leverage"].median()

    if "side" in df.columns:
        vc = df["side"].value_counts()
        result["long_count"] = vc.get("LONG", 0)
        result["short_count"] = vc.get("SHORT", 0)

    logger.info("EDA — total trades: %d | win rate: %.1f%%", result["total_trades"], result["win_rate"] * 100)
    return result


# ─────────────────────────────────────────────
# SENTIMENT VS PERFORMANCE
# ─────────────────────────────────────────────

def sentiment_performance_table(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate performance metrics by sentiment category."""
    available_sentiments = [s for s in SENTIMENT_ORDER if s in df["classification"].unique()]
    records = []
    for sent in available_sentiments:
        sub = df[df["classification"] == sent]
        pnl = sub["closedpnl"]
        rec: dict[str, Any] = {
            "sentiment": sent,
            "trade_count": len(sub),
            "total_pnl": pnl.sum(),
            "avg_pnl": pnl.mean(),
            "median_pnl": pnl.median(),
            "win_rate": (pnl > 0).mean() * 100,
            "pnl_std": pnl.std(),
            "max_profit": pnl.max(),
            "max_loss": pnl.min(),
        }
        if "leverage" in sub.columns:
            rec["avg_leverage"] = sub["leverage"].mean()
        if "is_long" in sub.columns:
            rec["long_ratio"] = sub["is_long"].mean() * 100
        if "risk_adjusted_return" in sub.columns:
            rar = sub["risk_adjusted_return"]
            rec["avg_risk_adj_return"] = rar.mean()
            rec["sharpe_like"] = rar.mean() / rar.std() if rar.std() > 0 else 0
        records.append(rec)

    tbl = pd.DataFrame(records)
    tbl.to_csv(PROCESSED_DIR / "sentiment_performance.csv", index=False)
    logger.info("Sentiment performance table:\n%s", tbl.to_string(index=False))
    return tbl


def run_statistical_tests(df: pd.DataFrame) -> dict[str, Any]:
    """Run ANOVA, pairwise t-tests, correlation, and effect size (Cohen's d)."""
    results: dict[str, Any] = {}
    groups = {s: df[df["classification"] == s]["closedpnl"].dropna() for s in SENTIMENT_ORDER
              if s in df["classification"].unique()}

    if len(groups) >= 2:
        f_stat, p_anova = stats.f_oneway(*groups.values())
        results["anova"] = {"f_statistic": float(f_stat), "p_value": float(p_anova),
                             "significant": bool(p_anova < 0.05)}
        logger.info("ANOVA PnL ~ Sentiment: F=%.3f, p=%.4f", f_stat, p_anova)

    # Pairwise t-tests
    ttests: list[dict] = []
    sent_list = list(groups.keys())
    for i in range(len(sent_list)):
        for j in range(i + 1, len(sent_list)):
            g1, g2 = groups[sent_list[i]], groups[sent_list[j]]
            t, p = stats.ttest_ind(g1, g2, equal_var=False)
            d = _cohens_d(g1, g2)
            ttests.append({"group1": sent_list[i], "group2": sent_list[j],
                           "t_stat": round(t, 4), "p_value": round(p, 4), "cohens_d": round(d, 4)})
    results["pairwise_ttests"] = ttests

    # Correlation: sentiment_score vs closedpnl
    if "sentiment_score" in df.columns:
        corr, p_corr = stats.pearsonr(df["sentiment_score"].dropna(), df.loc[df["sentiment_score"].notna(), "closedpnl"])
        results["correlation"] = {"pearson_r": round(corr, 4), "p_value": round(p_corr, 4)}
        logger.info("Pearson(sentiment_score, pnl): r=%.4f, p=%.4f", corr, p_corr)

    return results


def _cohens_d(a: pd.Series, b: pd.Series) -> float:
    """Compute Cohen's d effect size between two samples."""
    na, nb = len(a), len(b)
    pooled_std = np.sqrt(((na - 1) * a.std() ** 2 + (nb - 1) * b.std() ** 2) / (na + nb - 2))
    return (a.mean() - b.mean()) / pooled_std if pooled_std > 0 else 0.0


# ─────────────────────────────────────────────
# CLUSTERING
# ─────────────────────────────────────────────

def run_kmeans(feat_scaled: pd.DataFrame, n_clusters: int = 5) -> tuple[pd.Series, KMeans]:
    """Fit KMeans and return labels + fitted model."""
    np.random.seed(RANDOM_SEED)
    model = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=20)
    labels = pd.Series(model.fit_predict(feat_scaled), index=feat_scaled.index, name="kmeans_cluster")
    logger.info("KMeans inertia: %.2f", model.inertia_)
    return labels, model


def run_hierarchical(feat_scaled: pd.DataFrame) -> np.ndarray:
    """Compute hierarchical linkage matrix (Ward method)."""
    return linkage(feat_scaled.values, method="ward")


def label_clusters(ranking: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Attach cluster labels and assign human-readable segment names."""
    ranking = ranking.copy()
    ranking["cluster"] = labels.values

    # Compute per-cluster mean profile
    profile_cols = [c for c in ["account_total_pnl", "account_win_rate", "account_pnl_std"] if c in ranking.columns]
    profile = ranking.groupby("cluster")[profile_cols].mean()

    segment_map: dict[int, str] = {}
    for c in profile.index:
        row = profile.loc[c]
        pnl_hi = row.get("account_total_pnl", 0) > profile["account_total_pnl"].median()
        risk_hi = row.get("account_pnl_std", 0) > profile["account_pnl_std"].median()
        win_hi = row.get("account_win_rate", 0) > profile["account_win_rate"].median()
        if pnl_hi and not risk_hi:
            seg = "High Profit / Low Risk"
        elif pnl_hi and risk_hi:
            seg = "High Profit / High Risk"
        elif not pnl_hi and risk_hi:
            seg = "Low Profit / High Risk"
        elif win_hi:
            seg = "Consistent Traders"
        else:
            seg = "Losing Traders"
        segment_map[c] = seg

    ranking["segment"] = ranking["cluster"].map(segment_map)
    ranking.to_parquet(PROCESSED_DIR / "trader_clusters.parquet", index=False)
    return ranking


# ─────────────────────────────────────────────
# MACHINE LEARNING
# ─────────────────────────────────────────────

def run_classification(X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
    """Train classification models and return evaluation results."""
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1),
    }
    if XGB_AVAILABLE:
        models["XGBoost"] = xgb.XGBClassifier(n_estimators=200, random_state=RANDOM_SEED, eval_metric="logloss", verbosity=0)

    results: dict[str, Any] = {}
    for name, model in models.items():
        Xtr, Xte = (X_tr_s, X_te_s) if name == "Logistic Regression" else (X_tr, X_te)
        model.fit(Xtr, y_tr)
        pred = model.predict(Xte)
        prob = model.predict_proba(Xte)[:, 1] if hasattr(model, "predict_proba") else pred
        results[name] = {
            "accuracy": round(accuracy_score(y_te, pred), 4),
            "precision": round(precision_score(y_te, pred, zero_division=0), 4),
            "recall": round(recall_score(y_te, pred, zero_division=0), 4),
            "f1": round(f1_score(y_te, pred, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(y_te, prob), 4),
            "model": model,
            "feature_importance": _get_importance(model, X.columns.tolist()),
        }
        logger.info("%s — AUC=%.4f | F1=%.4f | Acc=%.4f", name, results[name]["roc_auc"], results[name]["f1"], results[name]["accuracy"])

    results["feature_names"] = X.columns.tolist()
    return results


def run_regression(X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
    """Train regression models and return evaluation results."""
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

    models = {
        "Random Forest Regressor": RandomForestRegressor(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, random_state=RANDOM_SEED),
    }

    results: dict[str, Any] = {}
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        results[name] = {
            "rmse": round(np.sqrt(mean_squared_error(y_te, pred)), 4),
            "mae": round(mean_absolute_error(y_te, pred), 4),
            "r2": round(r2_score(y_te, pred), 4),
            "model": model,
            "feature_importance": _get_importance(model, X.columns.tolist()),
        }
        logger.info("%s — RMSE=%.4f | MAE=%.4f | R²=%.4f", name, results[name]["rmse"], results[name]["mae"], results[name]["r2"])

    results["feature_names"] = X.columns.tolist()
    return results


def _get_importance(model: Any, feature_names: list[str]) -> Optional[pd.Series]:
    """Extract feature importances if available."""
    if hasattr(model, "feature_importances_"):
        return pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
    if hasattr(model, "coef_"):
        return pd.Series(np.abs(model.coef_[0]), index=feature_names).sort_values(ascending=False)
    return None
