"""
visualization.py
----------------
Publication-quality chart generation for the sentiment-trader project.
All figures are saved to outputs/figures/.

Author : Senior Quant Research Team
Version: 1.0.0
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.utils import FIGURES_DIR, SENTIMENT_COLOR_MAP, SENTIMENT_ORDER, get_logger

warnings.filterwarnings("ignore")
logger = get_logger(__name__)

# ── Global style ──────────────────────────────────────────────
STYLE = "dark_background"
PALETTE = [SENTIMENT_COLOR_MAP[s] for s in SENTIMENT_ORDER]
DPI = 150
FIG_W, FIG_H = 14, 7

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "text.color": "#c9d1d9",
    "axes.labelcolor": "#c9d1d9",
    "xtick.color": "#c9d1d9",
    "ytick.color": "#c9d1d9",
    "grid.color": "#21262d",
    "grid.alpha": 0.5,
})


def _save(fig: plt.Figure, name: str) -> Path:
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("Saved figure → %s", path)
    return path


def _sent_colors(labels: list[str]) -> list[str]:
    return [SENTIMENT_COLOR_MAP.get(l, "#888888") for l in labels]


# ── 1. Sentiment Distribution ─────────────────────────────────
def plot_sentiment_distribution(fg: pd.DataFrame) -> Path:
    counts = fg["classification"].value_counts().reindex(SENTIMENT_ORDER).dropna()
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(counts.index, counts.values, color=_sent_colors(counts.index.tolist()), edgecolor="#30363d", linewidth=0.8)
    ax.set_title("Bitcoin Sentiment Distribution", fontweight="bold", pad=15)
    ax.set_xlabel("Sentiment Class")
    ax.set_ylabel("Days Count")
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 2, f"{int(b.get_height()):,}", ha="center", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, "01_sentiment_distribution")


# ── 2. Daily Sentiment Timeline ───────────────────────────────
def plot_sentiment_timeline(fg: pd.DataFrame) -> Path:
    fg_sorted = fg.sort_values("date")
    fig, ax = plt.subplots(figsize=(FIG_W, 5))
    for sent in SENTIMENT_ORDER:
        mask = fg_sorted["classification"] == sent
        ax.scatter(fg_sorted.loc[mask, "date"], fg_sorted.loc[mask, "sentiment_score"],
                   label=sent, color=SENTIMENT_COLOR_MAP[sent], alpha=0.7, s=15, linewidths=0)
    ax.set_title("Bitcoin Sentiment Score Over Time", fontweight="bold", pad=15)
    ax.set_xlabel("Date")
    ax.set_ylabel("Sentiment Score (1–5)")
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(SENTIMENT_ORDER)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    return _save(fig, "02_sentiment_timeline")


# ── 3. PnL by Sentiment ───────────────────────────────────────
def plot_pnl_by_sentiment(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, FIG_H))
    order = [s for s in SENTIMENT_ORDER if s in df["classification"].unique()]
    colors = _sent_colors(order)

    # Box plot
    data_by_sent = [df.loc[df["classification"] == s, "closedpnl"].clip(-5000, 5000) for s in order]
    bp = axes[0].boxplot(data_by_sent, labels=order, patch_artist=True, notch=True,
                          medianprops=dict(color="white", linewidth=2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    axes[0].set_title("PnL Distribution by Sentiment (Box)", fontweight="bold")
    axes[0].set_xlabel("Sentiment")
    axes[0].set_ylabel("Closed PnL (USD)")
    axes[0].tick_params(axis="x", rotation=15)
    axes[0].grid(axis="y", alpha=0.3)

    # Mean PnL bar
    means = df.groupby("classification")["closedpnl"].mean().reindex(order)
    bars = axes[1].bar(order, means.values, color=colors, edgecolor="#30363d")
    for b, v in zip(bars, means.values):
        axes[1].text(b.get_x() + b.get_width() / 2, v + (abs(v) * 0.02), f"${v:.1f}", ha="center", fontsize=9)
    axes[1].axhline(0, color="white", linewidth=0.8, linestyle="--")
    axes[1].set_title("Average PnL per Trade by Sentiment", fontweight="bold")
    axes[1].set_xlabel("Sentiment")
    axes[1].set_ylabel("Mean PnL (USD)")
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle("PnL Analysis by Sentiment Category", fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    return _save(fig, "03_pnl_by_sentiment")


# ── 4. Win Rate by Sentiment ──────────────────────────────────
def plot_win_rate_by_sentiment(df: pd.DataFrame) -> Path:
    order = [s for s in SENTIMENT_ORDER if s in df["classification"].unique()]
    wr = df.groupby("classification")["is_win"].mean().reindex(order) * 100
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(order, wr.values, color=_sent_colors(order), edgecolor="#30363d")
    ax.axhline(50, color="white", linewidth=1, linestyle="--", label="50% baseline")
    for b, v in zip(bars, wr.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.1f}%", ha="center", fontsize=10)
    ax.set_title("Win Rate by Sentiment Category", fontweight="bold")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Win Rate (%)")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=15)
    return _save(fig, "04_win_rate_by_sentiment")


# ── 5. Leverage by Sentiment ──────────────────────────────────
def plot_leverage_by_sentiment(df: pd.DataFrame) -> Path:
    if "leverage" not in df.columns:
        logger.warning("No leverage column — skipping leverage chart.")
        return FIGURES_DIR / "05_leverage_by_sentiment.png"
    order = [s for s in SENTIMENT_ORDER if s in df["classification"].unique()]
    fig, ax = plt.subplots(figsize=(10, 6))
    data = [df.loc[df["classification"] == s, "leverage"].clip(0, 100) for s in order]
    vp = ax.violinplot(data, positions=range(len(order)), showmedians=True)
    for i, (body, color) in enumerate(zip(vp["bodies"], _sent_colors(order))):
        body.set_facecolor(color)
        body.set_alpha(0.7)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=15)
    ax.set_title("Leverage Distribution by Sentiment Category", fontweight="bold")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Leverage (x)")
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, "05_leverage_by_sentiment")


# ── 6. Trade Volume by Sentiment ──────────────────────────────
def plot_trade_volume_by_sentiment(df: pd.DataFrame) -> Path:
    order = [s for s in SENTIMENT_ORDER if s in df["classification"].unique()]
    vol = df.groupby("classification").size().reindex(order)
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(order, vol.values, color=_sent_colors(order), edgecolor="#30363d")
    for b, v in zip(bars, vol.values):
        ax.text(b.get_x() + b.get_width() / 2, v + vol.max() * 0.01, f"{int(v):,}", ha="center", fontsize=9)
    ax.set_title("Total Trade Count by Sentiment Category", fontweight="bold")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Number of Trades")
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=15)
    return _save(fig, "06_trade_volume_by_sentiment")


# ── 7. Correlation Heatmap ────────────────────────────────────
def plot_correlation_heatmap(df: pd.DataFrame) -> Path:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    drop_cols = [c for c in ["is_win", "is_long", "trade_direction"] if c in num_cols]
    feat_cols = [c for c in num_cols if c not in drop_cols][:20]
    corr = df[feat_cols].corr()
    fig, ax = plt.subplots(figsize=(14, 11))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, ax=ax, linewidths=0.5, cbar_kws={"shrink": 0.8},
                annot_kws={"size": 8})
    ax.set_title("Feature Correlation Heatmap", fontweight="bold", pad=15)
    fig.tight_layout()
    return _save(fig, "07_correlation_heatmap")


# ── 8. Trader Ranking Chart ───────────────────────────────────
def plot_trader_ranking(ranking: pd.DataFrame, top_n: int = 20) -> Path:
    top = ranking.head(top_n).copy()
    acct_labels = [str(a)[:8] + "…" if len(str(a)) > 10 else str(a) for a in top["account"]]
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.RdYlGn(np.linspace(0.9, 0.3, len(top)))
    bars = ax.barh(acct_labels[::-1], top["composite_score"].values[::-1], color=colors, edgecolor="#30363d")
    ax.set_title(f"Top {top_n} Traders by Composite Score", fontweight="bold")
    ax.set_xlabel("Composite Score (0–1)")
    ax.set_ylabel("Trader Account")
    ax.grid(axis="x", alpha=0.3)
    return _save(fig, "08_trader_ranking")


# ── 9. Cluster Visualization ──────────────────────────────────
def plot_clusters(ranking_with_clusters: pd.DataFrame, feat_scaled: pd.DataFrame) -> Path:
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(feat_scaled.values)
    coord_df = pd.DataFrame(coords, columns=["PC1", "PC2"], index=feat_scaled.index)
    plot_df = coord_df.join(ranking_with_clusters.set_index("account")[["segment"]])

    fig, ax = plt.subplots(figsize=(12, 8))
    palette_map = {s: c for s, c in zip(plot_df["segment"].unique(), plt.cm.Set2.colors)}
    for seg, grp in plot_df.groupby("segment"):
        ax.scatter(grp["PC1"], grp["PC2"], label=seg, alpha=0.75, s=60, color=palette_map.get(seg, "#888"))
    ax.set_title("Trader Cluster Visualization (PCA 2D)", fontweight="bold")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    return _save(fig, "09_cluster_visualization")


# ── 10. Feature Importance ────────────────────────────────────
def plot_feature_importance(importance: pd.Series, model_name: str, suffix: str = "") -> Path:
    top = importance.head(15)
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.viridis(np.linspace(0.8, 0.3, len(top)))
    ax.barh(top.index[::-1], top.values[::-1], color=colors, edgecolor="#30363d")
    ax.set_title(f"Feature Importance — {model_name}", fontweight="bold")
    ax.set_xlabel("Importance Score")
    ax.grid(axis="x", alpha=0.3)
    return _save(fig, f"10_feature_importance_{suffix}")


# ── 11. PnL Violin Plots ──────────────────────────────────────
def plot_pnl_violin(df: pd.DataFrame) -> Path:
    order = [s for s in SENTIMENT_ORDER if s in df["classification"].unique()]
    clipped = df.copy()
    clipped["closedpnl"] = clipped["closedpnl"].clip(-3000, 3000)
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    sns.violinplot(data=clipped, x="classification", y="closedpnl", order=order,
                   palette={s: SENTIMENT_COLOR_MAP[s] for s in order}, ax=ax,
                   inner="quartile", cut=0)
    ax.axhline(0, color="white", linewidth=0.8, linestyle="--")
    ax.set_title("PnL Violin Plot by Sentiment", fontweight="bold")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Closed PnL (USD, clipped ±3000)")
    ax.tick_params(axis="x", rotation=15)
    return _save(fig, "11_pnl_violin")


# ── 12. Daily PnL Time-Series ─────────────────────────────────
def plot_daily_pnl_timeseries(daily: pd.DataFrame) -> Path:
    daily_sorted = daily.sort_values("date")
    fig, axes = plt.subplots(2, 1, figsize=(FIG_W, 10), sharex=True)

    # Top: daily PnL bars coloured by sentiment
    colors = [SENTIMENT_COLOR_MAP.get(c, "#888") for c in daily_sorted["classification"]]
    axes[0].bar(daily_sorted["date"], daily_sorted["daily_pnl"], color=colors, alpha=0.85, width=0.9)
    axes[0].axhline(0, color="white", linewidth=0.8, linestyle="--")
    axes[0].set_title("Daily Total PnL (Coloured by Sentiment)", fontweight="bold")
    axes[0].set_ylabel("Daily PnL (USD)")
    axes[0].grid(alpha=0.3)

    # Bottom: sentiment score line
    axes[1].plot(daily_sorted["date"], daily_sorted["sentiment_score"], color="#1F77B4", linewidth=1.5)
    axes[1].fill_between(daily_sorted["date"], daily_sorted["sentiment_score"], alpha=0.2, color="#1F77B4")
    axes[1].set_title("Bitcoin Sentiment Score", fontweight="bold")
    axes[1].set_ylabel("Sentiment Score (1–5)")
    axes[1].set_xlabel("Date")
    axes[1].set_yticks([1, 2, 3, 4, 5])
    axes[1].grid(alpha=0.3)

    # Legend
    handles = [mpatches.Patch(color=SENTIMENT_COLOR_MAP[s], label=s) for s in SENTIMENT_ORDER]
    axes[0].legend(handles=handles, loc="upper left", fontsize=8)
    fig.tight_layout()
    return _save(fig, "12_daily_pnl_timeseries")


# ── 13. Long/Short Ratio by Sentiment ────────────────────────
def plot_long_short_by_sentiment(df: pd.DataFrame) -> Path:
    if "side" not in df.columns and "is_long" not in df.columns:
        logger.warning("No side/direction column — skipping long/short chart.")
        return FIGURES_DIR / "13_long_short_ratio.png"
    order = [s for s in SENTIMENT_ORDER if s in df["classification"].unique()]
    col = "is_long" if "is_long" in df.columns else None
    if col is None:
        df = df.copy()
        df["is_long"] = (df["side"] == "LONG").astype(int)
        col = "is_long"
    lr = df.groupby("classification")[col].mean().reindex(order) * 100

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(order, lr.values, color=_sent_colors(order), edgecolor="#30363d")
    ax.axhline(50, color="white", linewidth=1, linestyle="--", label="50% neutral")
    for b, v in zip(bars, lr.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.1f}%", ha="center", fontsize=10)
    ax.set_title("Long Trade Ratio by Sentiment Category", fontweight="bold")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("% Long Trades")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=15)
    return _save(fig, "13_long_short_ratio")


def run_all_visualizations(
    fg: pd.DataFrame,
    df: pd.DataFrame,
    daily: pd.DataFrame,
    ranking: Optional[pd.DataFrame] = None,
    ranking_clustered: Optional[pd.DataFrame] = None,
    feat_scaled: Optional[pd.DataFrame] = None,
    clf_results: Optional[dict] = None,
) -> dict[str, Path]:
    """Generate and save all figures, returning a dict of {name: path}."""
    paths: dict[str, Path] = {}
    paths["sentiment_dist"] = plot_sentiment_distribution(fg)
    paths["sentiment_timeline"] = plot_sentiment_timeline(fg)
    paths["pnl_by_sentiment"] = plot_pnl_by_sentiment(df)
    paths["win_rate"] = plot_win_rate_by_sentiment(df)
    paths["leverage"] = plot_leverage_by_sentiment(df)
    paths["trade_volume"] = plot_trade_volume_by_sentiment(df)
    paths["correlation"] = plot_correlation_heatmap(df)
    if ranking is not None and not ranking.empty:
        paths["trader_ranking"] = plot_trader_ranking(ranking)
    if ranking_clustered is not None and feat_scaled is not None:
        paths["clusters"] = plot_clusters(ranking_clustered, feat_scaled)
    if clf_results and "Random Forest" in clf_results:
        fi = clf_results["Random Forest"].get("feature_importance")
        if fi is not None:
            paths["feature_importance"] = plot_feature_importance(fi, "Random Forest", "rf_clf")
    paths["pnl_violin"] = plot_pnl_violin(df)
    paths["daily_pnl"] = plot_daily_pnl_timeseries(daily)
    paths["long_short"] = plot_long_short_by_sentiment(df)
    logger.info("All %d visualizations saved to %s", len(paths), FIGURES_DIR)
    return paths
