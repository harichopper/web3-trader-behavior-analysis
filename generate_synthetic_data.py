"""
generate_synthetic_data.py
--------------------------
Generate realistic synthetic datasets for the sentiment-trader project.
Used when real data files are not yet placed in data/.

Produces:
  data/fear_greed_index.csv         — 730 days of BTC sentiment
  data/hyperliquid_trades.csv       — ~50,000 synthetic trade records

Author : Senior Quant Research Team
Version: 1.0.0
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from src.utils import DATA_DIR, RANDOM_SEED, SENTIMENT_ORDER, get_logger

np.random.seed(RANDOM_SEED)
logger = get_logger("generate_data")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
N_DAYS = 730          # ~2 years of daily sentiment
N_TRADERS = 250       # unique trader accounts
N_TRADES = 55000      # total trade records
SYMBOLS = ["BTC-PERP", "ETH-PERP", "SOL-PERP", "ARB-PERP", "AVAX-PERP", "MATIC-PERP", "DOGE-PERP"]
LEVERAGE_DIST = [1, 2, 3, 5, 10, 20, 25, 50]


def generate_fear_greed(n_days: int = N_DAYS) -> pd.DataFrame:
    """Generate synthetic Bitcoin Fear & Greed Index data."""
    logger.info("Generating Fear & Greed dataset (%d days)…", n_days)
    end_date = pd.Timestamp("2024-12-31")
    start_date = end_date - pd.Timedelta(days=n_days - 1)
    dates = pd.date_range(start=start_date, end=end_date, freq="D")

    # Markov-chain transitions for realistic sentiment streaks
    transition = {
        "Extreme Fear": [0.40, 0.40, 0.10, 0.07, 0.03],
        "Fear":         [0.20, 0.35, 0.25, 0.15, 0.05],
        "Neutral":      [0.08, 0.22, 0.40, 0.22, 0.08],
        "Greed":        [0.05, 0.15, 0.25, 0.35, 0.20],
        "Extreme Greed":[0.03, 0.07, 0.10, 0.40, 0.40],
    }
    states: list[str] = ["Fear"]
    for _ in range(len(dates) - 1):
        probs = transition[states[-1]]
        states.append(np.random.choice(SENTIMENT_ORDER, p=probs))

    fg = pd.DataFrame({"date": dates, "classification": states})
    out = DATA_DIR / "fear_greed_index.csv"
    fg.to_csv(out, index=False)
    logger.info("Saved Fear & Greed → %s  (%d rows)", out, len(fg))
    return fg


def generate_trader_data(fg: pd.DataFrame, n_trades: int = N_TRADES, n_traders: int = N_TRADERS) -> pd.DataFrame:
    """Generate synthetic Hyperliquid trader data correlated with sentiment."""
    logger.info("Generating trader dataset (%d trades, %d traders)…", n_trades, n_traders)

    # Trader heterogeneity
    accounts = [f"0x{i:040x}" for i in range(1, n_traders + 1)]
    trader_skill = np.random.beta(2, 3, size=n_traders)          # win probability per trader
    trader_base_lev = np.random.choice([2, 5, 10, 20, 25, 50], size=n_traders, p=[0.10, 0.20, 0.30, 0.25, 0.10, 0.05])
    trader_long_bias = np.clip(np.random.normal(0.55, 0.15, size=n_traders), 0.1, 0.9)

    # Sentiment → PnL/behaviour mapping
    sent_pnl_mean  = {"Extreme Fear": -35, "Fear": -10, "Neutral": 5, "Greed": 25, "Extreme Greed": 50}
    sent_pnl_std   = {"Extreme Fear": 200,  "Fear": 150,  "Neutral": 120, "Greed": 180, "Extreme Greed": 250}
    sent_lev_mult  = {"Extreme Fear": 0.85, "Fear": 0.90, "Neutral": 1.00, "Greed": 1.15, "Extreme Greed": 1.35}
    sent_long_add  = {"Extreme Fear": -0.15, "Fear": -0.08, "Neutral": 0.0, "Greed": 0.08, "Extreme Greed": 0.15}

    rows = []
    for _ in range(n_trades):
        # Pick a random sentiment day
        day_idx = np.random.randint(0, len(fg))
        day = fg.iloc[day_idx]
        sent = day["classification"]
        date = pd.to_datetime(day["date"])

        # Pick trader
        t_idx = np.random.randint(0, n_traders)
        account = accounts[t_idx]
        skill = trader_skill[t_idx]
        base_lev = trader_base_lev[t_idx]
        long_bias = trader_long_bias[t_idx]

        # Leverage
        lev = max(1, base_lev * sent_lev_mult[sent] * np.random.uniform(0.7, 1.3))

        # Side
        long_prob = np.clip(long_bias + sent_long_add[sent], 0.05, 0.95)
        side = "LONG" if np.random.random() < long_prob else "SHORT"

        # PnL
        win = np.random.random() < (skill + (0.05 if sent in ("Greed", "Extreme Greed") else -0.05))
        pnl_mean = sent_pnl_mean[sent]
        pnl_std  = sent_pnl_std[sent]
        raw_pnl  = np.random.normal(pnl_mean, pnl_std) * (lev / 10)
        pnl = abs(raw_pnl) if win else -abs(raw_pnl)

        # Size
        symbol = np.random.choice(SYMBOLS)
        symbol_price = {"BTC-PERP": 45000, "ETH-PERP": 2500, "SOL-PERP": 120,
                         "ARB-PERP": 1.5, "AVAX-PERP": 35, "MATIC-PERP": 0.9, "DOGE-PERP": 0.12}
        price = symbol_price[symbol] * np.random.uniform(0.85, 1.15)
        size = np.random.exponential(0.5) * lev

        # Trade timestamp: random time on that day
        hour = np.random.randint(0, 24)
        minute = np.random.randint(0, 60)
        timestamp = date + pd.Timedelta(hours=int(hour), minutes=int(minute))
        ts_ms = int(timestamp.timestamp() * 1000)

        rows.append({
            "account": account,
            "symbol": symbol,
            "execution_price": round(price, 4),
            "size": round(size, 6),
            "side": side,
            "time": ts_ms,
            "closedPnL": round(pnl, 4),
            "leverage": round(lev, 2),
            "start_position": round(size * 0.5 * (-1 if side == "SHORT" else 1), 6),
            "event": np.random.choice(["close", "liquidation", "partial_close"], p=[0.88, 0.05, 0.07]),
        })

    trades = pd.DataFrame(rows)
    out = DATA_DIR / "hyperliquid_trades.csv"
    trades.to_csv(out, index=False)
    logger.info("Saved trader data → %s  (%d rows)", out, len(trades))
    return trades


if __name__ == "__main__":
    fg = generate_fear_greed()
    generate_trader_data(fg)
    logger.info("Synthetic data generation complete.")
