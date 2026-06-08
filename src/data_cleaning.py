"""
data_cleaning.py
----------------
End-to-end data loading, validation, cleaning, and merging pipeline for:
  • Bitcoin Fear & Greed Index dataset
  • Hyperliquid Historical Trader dataset

Author : Senior Quant Research Team
Version: 1.0.0
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.utils import (
    DATA_DIR,
    PROCESSED_DIR,
    SENTIMENT_ORDER,
    SENTIMENT_SCORE_MAP,
    get_logger,
    summarise_df,
)

warnings.filterwarnings("ignore")
logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────
# SECTION 1: FEAR & GREED INDEX LOADING
# ──────────────────────────────────────────────────────────────

def load_fear_greed(path: Optional[Path] = None) -> pd.DataFrame:
    """Load and clean the Bitcoin Fear & Greed Index dataset.

    Parameters
    ----------
    path:
        Optional explicit file path.  If *None*, scans ``data/`` for
        a CSV containing "fear" or "greed" in its filename (case-insensitive).

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with columns: ``date``, ``classification``,
        ``sentiment_score``.
    """
    if path is None:
        candidates = list(DATA_DIR.glob("*.csv")) + list(DATA_DIR.glob("*.xlsx"))
        matches = [
            p for p in candidates
            if any(kw in p.name.lower() for kw in ("fear", "greed", "sentiment", "fng"))
        ]
        # Fallback: pick whichever CSV contains a 'classification' column
        if not matches:
            for p in candidates:
                try:
                    peek = pd.read_csv(p, nrows=2)
                    if any(c.lower() in ("classification", "value_classification", "sentiment") for c in peek.columns):
                        matches = [p]
                        break
                except Exception:
                    continue
        if not matches:
            raise FileNotFoundError(
                "No Fear & Greed CSV found in data/. "
                "Expected a file with 'fear', 'greed', 'sentiment', 'fng' in the name, "
                "or a file containing a 'classification' column."
            )
        path = matches[0]
        logger.info("Auto-detected Fear & Greed file: %s", path.name)

    # ── Load ──────────────────────────────────────────────────
    ext = path.suffix.lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    logger.info("Raw Fear & Greed shape: %s", df.shape)
    summarise_df(df, "fear_greed_raw")

    # ── Normalise column names ─────────────────────────────────
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # ── Identify date column ───────────────────────────────────
    date_col = _find_column(df, ["date", "timestamp", "time", "day"])
    df = df.rename(columns={date_col: "date"})

    # ── Identify classification column ────────────────────────
    class_col = _find_column(df, ["classification", "value_classification", "sentiment", "label"])
    df = df.rename(columns={class_col: "classification"})

    # ── Parse dates ───────────────────────────────────────────
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["date"])
    df["date"] = df["date"].dt.normalize()  # strip time component → date only
    logger.info("Dropped %d rows with unparseable dates.", before - len(df))

    # ── Clean classification labels ────────────────────────────
    df["classification"] = (
        df["classification"]
        .astype(str)
        .str.strip()
        .str.title()
        .str.replace(r"\s+", " ", regex=True)
    )

    valid_labels = set(SENTIMENT_ORDER)
    invalid_mask = ~df["classification"].isin(valid_labels)
    if invalid_mask.any():
        logger.warning(
            "Found %d rows with unknown classification labels: %s — dropping.",
            invalid_mask.sum(),
            df.loc[invalid_mask, "classification"].unique().tolist(),
        )
        df = df[~invalid_mask]

    # ── Remove duplicates (keep latest entry per date) ─────────
    before = len(df)
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    logger.info("Removed %d duplicate date entries.", before - len(df))

    # ── Add ordinal sentiment score ────────────────────────────
    df["sentiment_score"] = df["classification"].map(SENTIMENT_SCORE_MAP)
    df["sentiment_category"] = pd.Categorical(
        df["classification"], categories=SENTIMENT_ORDER, ordered=True
    )

    df = df.sort_values("date").reset_index(drop=True)
    logger.info("Fear & Greed cleaned shape: %s", df.shape)
    return df[["date", "classification", "sentiment_score", "sentiment_category"]]


# ──────────────────────────────────────────────────────────────
# SECTION 2: HYPERLIQUID TRADER DATA LOADING
# ──────────────────────────────────────────────────────────────

def load_trader_data(path: Optional[Path] = None) -> pd.DataFrame:
    """Load and clean the Hyperliquid historical trader dataset.

    Parameters
    ----------
    path:
        Optional explicit file path.  Auto-detects from ``data/`` if *None*.

    Returns
    -------
    pd.DataFrame
        Cleaned and standardised trader DataFrame.
    """
    if path is None:
        candidates = list(DATA_DIR.glob("*.csv")) + list(DATA_DIR.glob("*.xlsx"))
        # First pass: keyword match
        matches = [
            p for p in candidates
            if any(
                kw in p.name.lower()
                for kw in ("hyperliquid", "trader", "trade", "hl_", "positions", "fills", "historical")
            )
        ]
        # Second pass: exclude any sentiment file, take the largest remaining CSV
        if not matches:
            sentiment_files = {
                p for p in candidates
                if any(kw in p.name.lower() for kw in ("fear", "greed", "sentiment", "fng"))
            }
            non_sentiment = [p for p in candidates if p not in sentiment_files]
            if non_sentiment:
                matches = [max(non_sentiment, key=lambda p: p.stat().st_size)]
        if not matches:
            raise FileNotFoundError(
                "No Hyperliquid trader CSV found in data/. "
                "Expected a file with 'hyperliquid', 'trader', 'trade', 'historical', or 'fills' in the name."
            )
        path = matches[0]
        logger.info("Auto-detected trader file: %s", path.name)

    ext = path.suffix.lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception:
            df = pd.read_csv(path, low_memory=False, encoding="latin-1")

    logger.info("Raw trader shape: %s", df.shape)
    summarise_df(df, "trader_raw")

    # ── Normalise column names ─────────────────────────────────
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    col_map = _build_column_map(df)
    df = df.rename(columns=col_map)
    logger.info("Mapped columns: %s", col_map)

    # ── Parse timestamp ────────────────────────────────────────
    df = _parse_trade_time(df)

    # ── Numeric coercion ──────────────────────────────────────
    for col in ["closedpnl", "size", "execution_price", "leverage"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Remove rows missing critical fields ───────────────────
    critical = [c for c in ["account", "time", "closedpnl"] if c in df.columns]
    before = len(df)
    df = df.dropna(subset=critical)
    logger.info("Dropped %d rows missing critical fields: %s", before - len(df), critical)

    # ── Remove duplicate trades ───────────────────────────────
    before = len(df)
    dup_cols = [c for c in ["account", "time", "symbol", "size", "closedpnl"] if c in df.columns]
    df = df.drop_duplicates(subset=dup_cols, keep="first")
    logger.info("Removed %d duplicate trade rows.", before - len(df))

    # ── Standardise side ──────────────────────────────────────
    # Prefer 'direction' column (has Open Long / Close Short etc.) over 'side' (BUY/SELL)
    # because direction gives more semantic meaning for position classification
    dir_col = None
    if "direction" in df.columns:
        dir_col = "direction"
    elif "side" in df.columns:
        dir_col = "side"

    if dir_col:
        raw = df[dir_col].astype(str).str.strip()
        # Map all known Hyperliquid direction variants
        side_map = {
            "BUY": "LONG", "SELL": "SHORT",
            "B": "LONG",   "S": "SHORT",
            "1": "LONG",   "-1": "SHORT",
            "OPEN LONG": "LONG",  "CLOSE LONG": "LONG",
            "OPEN SHORT": "SHORT", "CLOSE SHORT": "SHORT",
            "LONG > SHORT": "SHORT", "SHORT > LONG": "LONG",
            "BUY": "LONG", "SELL": "SHORT",
        }
        df["side"] = raw.str.upper().map(side_map)
        # Fallback: if still unmapped, try partial match
        unmapped = df["side"].isna()
        if unmapped.any():
            df.loc[unmapped & raw.str.upper().str.contains("LONG", na=False), "side"] = "LONG"
            df.loc[unmapped & raw.str.upper().str.contains("SHORT", na=False), "side"] = "SHORT"

    # ── Add trade date ─────────────────────────────────────────
    df["date"] = df["time"].dt.normalize()

    # ── Add trade direction flag ──────────────────────────────
    if "side" in df.columns:
        df["is_long"] = (df["side"] == "LONG").astype(int)

    # ── Add win flag ──────────────────────────────────────────
    if "closedpnl" in df.columns:
        df["is_win"] = (df["closedpnl"] > 0).astype(int)

    df = df.sort_values("time").reset_index(drop=True)
    logger.info("Trader data cleaned shape: %s", df.shape)
    return df


# ──────────────────────────────────────────────────────────────
# SECTION 3: MERGING
# ──────────────────────────────────────────────────────────────

def merge_datasets(
    fg: pd.DataFrame,
    trades: pd.DataFrame,
) -> pd.DataFrame:
    """Merge fear-greed and trader datasets on trading date.

    Parameters
    ----------
    fg:
        Cleaned fear & greed DataFrame (output of ``load_fear_greed``).
    trades:
        Cleaned trader DataFrame (output of ``load_trader_data``).

    Returns
    -------
    pd.DataFrame
        Merged DataFrame where every trade row is annotated with the
        corresponding day's Bitcoin sentiment.
    """
    # Ensure both date columns are timezone-naive date-only datetimes
    fg["date"] = pd.to_datetime(fg["date"]).dt.normalize().dt.tz_localize(None)
    trades["date"] = pd.to_datetime(trades["date"]).dt.normalize().dt.tz_localize(None)

    merged = trades.merge(
        fg[["date", "classification", "sentiment_score", "sentiment_category"]],
        on="date",
        how="inner",
        validate="m:1",
    )

    logger.info(
        "Merged dataset: %d trades over %d unique dates.",
        len(merged),
        merged["date"].nunique(),
    )
    coverage = len(merged) / len(trades) * 100
    logger.info("Sentiment coverage: %.1f%% of trades matched.", coverage)

    # Save to disk
    out = PROCESSED_DIR / "merged_dataset.parquet"
    merged.to_parquet(out, index=False)
    logger.info("Saved merged dataset → %s", out)

    return merged.reset_index(drop=True)


# ──────────────────────────────────────────────────────────────
# SECTION 4: DAILY AGGREGATION
# ──────────────────────────────────────────────────────────────

def build_daily_aggregates(merged: pd.DataFrame) -> pd.DataFrame:
    """Compute daily aggregated statistics from the merged dataset.

    Returns
    -------
    pd.DataFrame
        One row per date with aggregate PnL, volume, counts, etc.
    """
    agg_dict: dict = {
        "closedpnl": ["sum", "mean", "median", "std", "count"],
        "is_win": "mean",
    }
    if "size" in merged.columns:
        agg_dict["size"] = "sum"
    if "leverage" in merged.columns:
        agg_dict["leverage"] = "mean"
    if "is_long" in merged.columns:
        agg_dict["is_long"] = "mean"

    daily = merged.groupby(["date", "classification", "sentiment_score"]).agg(agg_dict)
    daily.columns = ["_".join(c).strip("_") for c in daily.columns]
    daily = daily.rename(
        columns={
            "closedpnl_sum": "daily_pnl",
            "closedpnl_mean": "avg_pnl_per_trade",
            "closedpnl_median": "median_pnl",
            "closedpnl_std": "pnl_std",
            "closedpnl_count": "trade_count",
            "is_win_mean": "win_rate",
            "size_sum": "daily_volume",
            "leverage_mean": "avg_leverage",
            "is_long_mean": "long_ratio",
        }
    )
    daily = daily.reset_index().sort_values("date")

    out = PROCESSED_DIR / "daily_aggregates.parquet"
    daily.to_parquet(out, index=False)
    logger.info("Saved daily aggregates → %s", out)
    return daily


# ──────────────────────────────────────────────────────────────
# SECTION 5: PREPROCESSING SUMMARY
# ──────────────────────────────────────────────────────────────

def generate_preprocessing_summary(
    fg_raw_shape: tuple,
    trader_raw_shape: tuple,
    fg_clean: pd.DataFrame,
    trader_clean: pd.DataFrame,
    merged: pd.DataFrame,
) -> str:
    """Generate a markdown-formatted preprocessing summary string."""
    lines = [
        "# Data Preprocessing Summary\n",
        "## Fear & Greed Index",
        f"- Raw rows: {fg_raw_shape[0]:,}",
        f"- Cleaned rows: {len(fg_clean):,}",
        f"- Date range: {fg_clean['date'].min().date()} → {fg_clean['date'].max().date()}",
        f"- Sentiment classes: {sorted(fg_clean['classification'].unique())}",
        "",
        "## Hyperliquid Trader Data",
        f"- Raw rows: {trader_raw_shape[0]:,}",
        f"- Cleaned rows: {len(trader_clean):,}",
        f"- Unique accounts: {trader_clean['account'].nunique():,}" if "account" in trader_clean.columns else "",
        f"- Unique symbols: {trader_clean['symbol'].nunique():,}" if "symbol" in trader_clean.columns else "",
        f"- Date range: {trader_clean['date'].min().date()} → {trader_clean['date'].max().date()}",
        "",
        "## Merged Dataset",
        f"- Total rows: {len(merged):,}",
        f"- Unique dates: {merged['date'].nunique():,}",
        f"- Sentiment coverage: {len(merged)/max(1,len(trader_clean))*100:.1f}%",
        f"- Sentiment breakdown:\n{merged['classification'].value_counts().to_string()}",
    ]
    summary = "\n".join(l for l in lines if l is not None)
    logger.info("\n%s", summary)
    return summary


# ──────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ──────────────────────────────────────────────────────────────

def _find_column(df: pd.DataFrame, candidates: list[str]) -> str:
    """Return the first column name from *candidates* found in *df*.

    Raises ``KeyError`` if none found.
    """
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        f"Could not find any of {candidates} in columns {list(df.columns)}"
    )


def _build_column_map(df: pd.DataFrame) -> dict[str, str]:
    """Build a column rename map for known Hyperliquid field aliases."""
    alias: dict[str, list[str]] = {
        "account": ["account", "user", "wallet", "trader", "address"],
        # 'coin' is the Hyperliquid historical export column name for symbol
        "symbol": ["symbol", "coin", "asset", "pair", "market"],
        # 'execution_price' maps from 'execution price' (space normalised to _)
        "execution_price": ["execution_price", "px", "price", "avg_price", "fill_price"],
        # 'size_tokens' or 'size_usd' — prefer size_tokens as base unit
        "size": ["size", "size_tokens", "sz", "quantity", "qty", "amount"],
        "side": ["side", "trade_side"],
        # 'timestamp' (unix ms) preferred over 'timestamp_ist' (string)
        "time": ["time", "timestamp", "ts", "created_at", "datetime", "trade_time"],
        "start_position": ["start_position", "start_pos", "startposition"],
        # 'closed_pnl' → Hyperliquid export column 'closed pnl' (space → _)
        "closedpnl": ["closedpnl", "closed_pnl", "pnl", "realized_pnl", "realizedpnl", "profit"],
        "leverage": ["leverage", "lev", "leverage_used"],
        "direction": ["direction", "event", "event_type", "type"],
        "fee": ["fee"],
        "size_usd": ["size_usd"],
    }
    col_map: dict[str, str] = {}
    existing = set(df.columns)
    for canonical, aliases in alias.items():
        if canonical in existing:
            continue  # already named correctly
        for a in aliases:
            if a in existing:
                col_map[a] = canonical
                break
    return col_map


def _parse_trade_time(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the ``time`` column to timezone-aware UTC datetime."""
    if "time" not in df.columns:
        raise KeyError("Expected a 'time' column after column normalisation.")

    series = df["time"]

    # If numeric (Unix ms or Unix s), convert
    if pd.api.types.is_numeric_dtype(series):
        if series.max() > 1e12:
            df["time"] = pd.to_datetime(series, unit="ms", utc=True)
        else:
            df["time"] = pd.to_datetime(series, unit="s", utc=True)
    else:
        df["time"] = pd.to_datetime(series, errors="coerce", utc=True)

    # Localise to UTC then strip tz for consistency
    df["time"] = df["time"].dt.tz_convert("UTC").dt.tz_localize(None)
    return df
