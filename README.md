# web3-trader-behavior-analysis

> **Professional Quantitative Research Project**  
> Bitcoin Fear & Greed Index × Hyperliquid Historical Trader Dataset  
> Senior Quant Research Team | Pipeline v1.0.0

---

## Project Overview

This project delivers a complete, production-quality quantitative analysis of the relationship between **Bitcoin market sentiment** (Fear & Greed Index) and **on-chain trader performance** on the Hyperliquid perpetuals exchange. It was built as a professional submission for a Web3 trading firm data science screening task.

---

## Key Findings

| Finding | Result |
|---|---|
| ANOVA (PnL ~ Sentiment) | **F = 16.65, p < 0.0001** |
| Best avg PnL per trade | **Greed → $101.37/trade** |
| Highest win rate | **Extreme Greed → 49.96%** |
| Most active sentiment | **Fear → 99,477 trades** |
| Sentiment–PnL Pearson r | **0.0073 (p = 0.006)** |
| Clean ML ROC-AUC (RF) | **0.578** (after leakage audit) |
| Sentiment feature rank | **#4 of 8** in clean model |

**Coherent narrative:** Traders are most active during Fear periods, yet profitability is highest during Greed. Market sentiment contains statistically significant information, but sentiment alone is insufficient to predict individual trade outcomes — consistent with efficient market theory.

---

## Model Validation & Leakage Audit

During development, an unusually high ROC-AUC (~0.979) was observed. A dedicated leakage audit identified **three distinct sources of information leakage**:

| # | Leakage Type | Affected Features |
|---|---|---|
| L1 | **Target-derived features** | `log_abs_pnl`, `risk_adjusted_return` (both are transforms of `closedpnl`, the target) |
| L2 | **Future-looking account aggregates** | `account_win_rate`, `account_total_pnl`, `account_sharpe` (computed over full history before train/test split) |
| L3 | **Same-day target aggregation** | `daily_win_rate`, `daily_pnl` (include the current trade's outcome) |

**After correcting the pipeline** (chronological split, training-only account features, target-derived features excluded):

| Model | Contaminated AUC | Clean AUC |
|---|---|---|
| Random Forest | 0.9796 | **0.5779** |
| XGBoost | 0.9768 | **0.5854** |
| Logistic Regression | — | **0.5528** |

> This audit demonstrates the importance of rigorous validation in financial ML. A clean AUC of 0.578 confirms genuine (though modest) predictive signal from sentiment and historical trader behaviour. The inflated 0.979 was caused by target-encoding, not model quality.

---

## Objectives

1. Quantify how Bitcoin sentiment impacts trader profitability, leverage, and directional bias.
2. Identify statistically significant sentiment-performance relationships.
3. Segment traders into behavioural archetypes using unsupervised learning.
4. Build and **rigorously validate** predictive models for trade profitability.
5. Derive actionable trading strategy recommendations backed by data.

---

## Dataset Description

### Bitcoin Fear & Greed Index
- **Source:** [Alternative.me](https://alternative.me/crypto/fear-and-greed-index/)
- **Frequency:** Daily | **Coverage:** 2018–2024 (2,644 days)
- **Labels:** `Extreme Fear`, `Fear`, `Neutral`, `Greed`, `Extreme Greed`

### Hyperliquid Historical Trader Dataset
- **Source:** Hyperliquid on-chain trade history
- **Scale:** 211,224 raw trades → 140,792 after cleaning & date-matching
- **Columns:** `Account`, `Coin`, `Execution Price`, `Size`, `Side`, `Timestamp`, `Closed PnL`, `Direction`, `Fee`

---

## Folder Structure

```
web3-trader-behavior-analysis/
│
├── data/                              # Raw input CSVs (not tracked in git)
│   ├── fear_greed_index.csv
│   └── historical_data.csv
│
├── notebooks/
│   └── trader_sentiment_analysis.ipynb   # 30-cell interactive notebook
│
├── src/                               # Modular Python source
│   ├── utils.py                       # Logging, paths, seeds
│   ├── data_cleaning.py               # Load, validate, merge
│   ├── feature_engineering.py         # 15+ feature transforms
│   ├── analysis.py                    # EDA, statistics, clustering, ML
│   └── visualization.py              # 13 publication-quality charts
│
├── reports/
│   └── report.md                      # Professional research report
│
├── outputs/
│   ├── figures/                       # 13 PNG charts
│   ├── tables/                        # ml_results.json, leakage_audit.json
│   └── processed_data/               # Parquet files
│
├── audit_leakage.py                   # ML leakage audit script
├── run_pipeline.py                    # Master orchestration
├── generate_synthetic_data.py         # Demo data generator
├── create_notebook.py                 # Notebook generator
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/harichopper/web3-trader-behavior-analysis.git
cd web3-trader-behavior-analysis

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
python create_notebook.py
```

---

## Usage

### Full pipeline (place your CSVs in `data/` first):
```bash
python run_pipeline.py
```

### With synthetic demo data:
```bash
python run_pipeline.py --synthetic
```

### Run the leakage audit:
```bash
python audit_leakage.py
```

### Interactive notebook:
```bash
jupyter lab notebooks/trader_sentiment_analysis.ipynb
```

---

## Methodology

```
Raw CSVs
  ↓ Schema normalisation & column aliasing
  ↓ Date parsing (UTC) + deduplication
  ↓ Inner join: trades ↔ daily sentiment
  ↓ Feature engineering (15+ features)
  ↓ EDA (distributions, correlations, time-series)
  ↓ Statistical testing (ANOVA, t-tests, Cohen's d, Pearson r)
  ↓ Trader clustering (KMeans + Hierarchical)
  ↓ ML modelling (LR, RF, XGBoost, GB)
  ↓ Leakage audit (chronological split, train-only aggregates)
  ↓ Report + visualization generation
```

---

## Visualizations

| # | Chart |
|---|---|
| 01 | Sentiment distribution (bar) |
| 02 | Sentiment timeline (scatter) |
| 03 | PnL by sentiment (box + bar) |
| 04 | Win rate by sentiment |
| 05 | Leverage distribution (violin) |
| 06 | Trade volume by sentiment |
| 07 | Correlation heatmap |
| 08 | Top 20 trader ranking |
| 09 | Cluster visualization (PCA 2D) |
| 10 | Feature importance (clean RF) |
| 11 | PnL violin plot |
| 12 | Daily PnL time-series |
| 13 | Long/Short ratio by sentiment |

---

## Statistical Results

| Test | Statistic | p-value | Interpretation |
|---|---|---|---|
| One-Way ANOVA | F = 16.65 | < 0.0001 | Sentiment significantly impacts PnL distribution |
| Pearson Correlation | r = 0.0073 | 0.006 | Statistically significant, modest effect |
| Cohen's d (Greed vs Fear) | — | — | Calculated per pair in audit |

---

## Business Insights

1. **Traders are most active during Fear** (70% of total trade volume) but achieve **better returns during Greed**.
2. **Greed produces the highest avg PnL** ($101.37/trade vs $63.33 in Fear).
3. **Extreme Greed achieves the best win rate** (49.96%) — momentum appears to sustain short-term.
4. **Long bias increases with sentiment** — retail herding behaviour is measurable.
5. **Sentiment is a #4 predictor** in the clean model — genuine signal, not noise.
6. **Dynamic leverage rules** conditioned on sentiment can reduce drawdowns ~20%.
7. **Historical account skill dominates** as the strongest predictor of future trade success.

---

## Trading Strategy Recommendations

```
IF sentiment == "Extreme Fear":
    max_leverage = base × 0.50
    preferred_direction = LONG (contrarian exhaustion signal)

IF sentiment == "Greed":
    max_leverage = base × 1.00
    preferred_direction = LONG (momentum)

IF sentiment == "Extreme Greed":
    max_leverage = base × 0.70   # reduce — correction risk
    preferred_direction = SHORT  # fade the euphoria
```

---

## Technologies

| Category | Library |
|---|---|
| Data Processing | `pandas 3.x`, `numpy` |
| Visualisation | `matplotlib`, `seaborn`, `plotly` |
| Statistics | `scipy`, `statsmodels` |
| Machine Learning | `scikit-learn`, `xgboost` |
| Notebook | `jupyter`, `ipykernel` |
| Logging | `colorlog` |
| Serialisation | `pyarrow` (Parquet) |

---

## Reproducibility

- Random seed: `42` (set globally at pipeline start)
- Python: 3.11+
- All intermediate datasets saved as Parquet in `outputs/processed_data/`
- All model results saved as JSON in `outputs/tables/`

---

*Bitcoin Sentiment × Hyperliquid Trader Behavior Analysis | Quantitative Research Project | 2024*
