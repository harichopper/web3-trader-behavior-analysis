"""
utils.py
--------
Shared utility functions: logging, path management, configuration,
reproducibility helpers, and formatting shortcuts.

Author : Senior Quant Research Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import colorlog

# ─────────────────────────────────────────────
# Project root / output paths
# ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"
PROCESSED_DIR = OUTPUTS_DIR / "processed_data"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Ensure directories exist
for _d in (DATA_DIR, FIGURES_DIR, TABLES_DIR, PROCESSED_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────
RANDOM_SEED: int = 42


def set_random_seeds(seed: int = RANDOM_SEED) -> None:
    """Fix all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import sklearn  # noqa: F401
    except ImportError:
        pass


# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
_LOG_FORMAT = "%(log_color)s%(asctime)s [%(levelname)-8s] %(name)s — %(message)s%(reset)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_COLOR_MAP = {
    "DEBUG": "cyan",
    "INFO": "green",
    "WARNING": "yellow",
    "ERROR": "red",
    "CRITICAL": "bold_red",
}


def get_logger(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    """Return a colour-formatted logger instance.

    Parameters
    ----------
    name:
        Logger name (use ``__name__`` in each module).
    level:
        Logging level (default: INFO).

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    handler = colorlog.StreamHandler(stream=sys.stdout)
    handler.setFormatter(
        colorlog.ColoredFormatter(
            fmt=_LOG_FORMAT,
            datefmt=_DATE_FORMAT,
            log_colors=_COLOR_MAP,
        )
    )
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


# ─────────────────────────────────────────────
# Sentiment helpers
# ─────────────────────────────────────────────
SENTIMENT_ORDER: list[str] = [
    "Extreme Fear",
    "Fear",
    "Neutral",
    "Greed",
    "Extreme Greed",
]

SENTIMENT_SCORE_MAP: dict[str, int] = {
    "Extreme Fear": 1,
    "Fear": 2,
    "Neutral": 3,
    "Greed": 4,
    "Extreme Greed": 5,
}

SENTIMENT_COLOR_MAP: dict[str, str] = {
    "Extreme Fear": "#D62728",
    "Fear": "#FF7F0E",
    "Neutral": "#BCBD22",
    "Greed": "#2CA02C",
    "Extreme Greed": "#1F77B4",
}


def sentiment_to_score(label: str) -> int:
    """Map a sentiment label to an ordinal integer score (1–5)."""
    return SENTIMENT_SCORE_MAP.get(label, 0)


# ─────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────

def fmt_pct(value: float, decimals: int = 2) -> str:
    """Format a ratio as a percentage string."""
    return f"{value * 100:.{decimals}f}%"


def fmt_money(value: float, symbol: str = "$") -> str:
    """Format a float as a dollar amount."""
    return f"{symbol}{value:,.2f}"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Division with zero-denominator guard."""
    if denominator == 0:
        return default
    return numerator / denominator


def print_section(title: str, width: int = 70) -> None:
    """Print a formatted section header to stdout."""
    border = "=" * width
    print(f"\n{border}")
    print(f"  {title.upper()}")
    print(f"{border}\n")


# ─────────────────────────────────────────────
# DataFrame helpers
# ─────────────────────────────────────────────

def summarise_df(df: Any, name: str = "DataFrame") -> None:  # noqa: ANN401
    """Print a concise summary of a DataFrame."""
    logger = get_logger("utils")
    logger.info("=== %s ===", name)
    logger.info("Shape    : %s rows × %s cols", *df.shape)
    logger.info("Columns  : %s", list(df.columns))
    nulls = df.isnull().sum()
    if nulls.any():
        logger.warning("Nulls:\n%s", nulls[nulls > 0])
    else:
        logger.info("Nulls    : none")
    dups = df.duplicated().sum()
    if dups:
        logger.warning("Duplicates: %d rows", dups)
    else:
        logger.info("Duplicates: none")
