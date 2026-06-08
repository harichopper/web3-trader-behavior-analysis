"""
feature_engineering.py
-----------------------
All feature engineering transforms applied to the merged dataset.

Author : Senior Quant Research Team
Version: 1.0.0
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import PROCESSED_DIR, SENTIMENT_SCORE_MAP, get_logger, safe_divide

logger = get_logger(__name__)


def add_trade_level_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-trade engineered features."""
    logger.info("Adding trade-level features …")
    df = df.copy()
    df["pnl_per_trade"] = df["closedpnl"]

    if "leverage" in df.columns:
        bins = [0, 2, 5, 10, 20, 50, np.inf]
        labels = ["1-2x", "2-5x", "5-10x", "10-20x", "20-50x", "50x+"]
        df["leverage_bucket"] = pd.cut(df["leverage"].clip(lower=0), bins=bins, labels=labels, right=True)
    else:
        df["leverage_bucket"] = "unknown"

    if "sentiment_score" not in df.columns:
        df["sentiment_score"] = df["classification"].map(SENTIMENT_SCORE_MAP).fillna(0)

    if "side" in df.columns:
        df["trade_direction"] = df["side"].map({"LONG": 1, "SHORT": -1}).fillna(0)
    elif "is_long" in df.columns:
        df["trade_direction"] = df["is_long"].map({1: 1, 0: -1})
    else:
        df["trade_direction"] = 0

    if "leverage" in df.columns:
        lev = df["leverage"].replace(0, np.nan).fillna(1)
        df["risk_adjusted_return"] = df["closedpnl"] / lev
    else:
        df["risk_adjusted_return"] = df["closedpnl"]

    df["log_abs_pnl"] = np.log1p(df["closedpnl"].abs())

    if "execution_price" in df.columns and "size" in df.columns:
        df["trade_size_usd"] = df["execution_price"] * df["size"].abs()

    return df


def add_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add daily aggregate features back onto each trade row."""
    logger.info("Adding daily aggregate features …")
    daily = (
        df.groupby("date")
        .agg(daily_trade_count=("closedpnl", "count"), daily_pnl=("closedpnl", "sum"), daily_win_rate=("is_win", "mean"))
        .reset_index()
    )
    return df.merge(daily, on="date", how="left")


def add_account_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-account aggregate statistics and merge back."""
    if "account" not in df.columns:
        logger.warning("No 'account' column — skipping account features.")
        return df
    logger.info("Computing account-level features …")

    def sharpe_like(s: pd.Series) -> float:
        std = s.std()
        return safe_divide(s.mean(), std) if std > 0 else 0.0

    def max_drawdown(s: pd.Series) -> float:
        cumulative = s.cumsum()
        return (cumulative.cummax() - cumulative).max()

    agg = {"account_total_pnl": ("closedpnl", "sum"), "account_trade_count": ("closedpnl", "count"),
           "account_win_rate": ("is_win", "mean"), "account_pnl_std": ("closedpnl", "std")}
    if "leverage" in df.columns:
        agg["account_avg_leverage"] = ("leverage", "mean")
    if "is_long" in df.columns:
        agg["account_long_ratio"] = ("is_long", "mean")

    acct_agg = df.groupby("account").agg(**agg).reset_index()
    sharpe = df.groupby("account")["closedpnl"].apply(sharpe_like).rename("account_sharpe")
    mdd = df.groupby("account")["closedpnl"].apply(max_drawdown).rename("account_max_drawdown")
    acct_agg = acct_agg.merge(sharpe, on="account").merge(mdd, on="account")
    return df.merge(acct_agg, on="account", how="left")


def rank_traders(df: pd.DataFrame) -> pd.DataFrame:
    """Compute a composite trader ranking."""
    if "account" not in df.columns:
        return pd.DataFrame()

    cols = ["account", "account_total_pnl", "account_win_rate", "account_sharpe",
            "account_trade_count", "account_pnl_std"]
    cols += [c for c in ["account_avg_leverage", "account_long_ratio", "account_max_drawdown"] if c in df.columns]
    ranking = df[cols].drop_duplicates("account").copy()

    def _minmax(s: pd.Series) -> pd.Series:
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng != 0 else pd.Series(0.5, index=s.index)

    ranking["norm_pnl"] = _minmax(ranking["account_total_pnl"])
    ranking["norm_win_rate"] = _minmax(ranking["account_win_rate"])
    ranking["norm_sharpe"] = _minmax(ranking["account_sharpe"])
    ranking["composite_score"] = 0.40 * ranking["norm_pnl"] + 0.35 * ranking["norm_win_rate"] + 0.25 * ranking["norm_sharpe"]
    ranking["composite_rank"] = ranking["composite_score"].rank(ascending=False, method="min").astype(int)
    ranking = ranking.sort_values("composite_rank")
    ranking.to_parquet(PROCESSED_DIR / "trader_ranking.parquet", index=False)
    return ranking


def build_clustering_matrix(ranking: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Select and z-score scale features for clustering."""
    from scipy.stats import zscore as _zscore
    feature_cols = [c for c in ["account_total_pnl", "account_win_rate", "account_sharpe",
                                 "account_trade_count", "account_pnl_std", "account_avg_leverage",
                                 "account_long_ratio"] if c in ranking.columns]
    feat_df = ranking.set_index("account")[feature_cols].fillna(0)
    feat_scaled = feat_df.apply(_zscore, nan_policy="omit").fillna(0)
    feat_scaled.columns = feature_cols
    return feat_scaled, feature_cols


def build_ml_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Build feature matrix X and targets for ML models."""
    candidate = ["sentiment_score", "trade_direction", "leverage", "log_abs_pnl",
                 "daily_trade_count", "daily_win_rate", "account_win_rate",
                 "account_total_pnl", "account_sharpe", "account_avg_leverage",
                 "account_long_ratio", "account_trade_count"]
    if "trade_size_usd" in df.columns:
        candidate.append("trade_size_usd")
    feature_cols = [c for c in candidate if c in df.columns]
    sub = df[feature_cols + ["is_win", "pnl_per_trade"]].dropna()
    return sub[feature_cols], sub["is_win"].astype(int), sub["pnl_per_trade"]
