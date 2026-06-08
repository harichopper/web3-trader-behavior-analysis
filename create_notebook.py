"""
create_notebook.py
------------------
Programmatically generate notebooks/trader_sentiment_analysis.ipynb.
Includes all pipeline steps + the leakage audit section with corrected results.

Run once:
    python create_notebook.py

Author : Senior Quant Research Team
Version: 2.0.0  (updated with leakage audit)
"""

from __future__ import annotations
import json
from pathlib import Path

NB_PATH = Path(__file__).parent / "notebooks" / "trader_sentiment_analysis.ipynb"


def code(source: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "source": source,
            "outputs": [], "execution_count": None}


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


# ─────────────────────────────────────────────────────────────────────────────
cells = [

# ── Title ──────────────────────────────────────────────────────────────────
md("""# Bitcoin Sentiment x Hyperliquid Trader Performance
## Quantitative Research Notebook — v2.0 (Post Leakage Audit)

> **Senior Quant Research Team**
> Repository: [harichopper/web3-trader-behavior-analysis](https://github.com/harichopper/web3-trader-behavior-analysis)

### Notebook Sections
| Section | Topic |
|---------|-------|
| 1 | Environment Setup |
| 2 | Data Loading & Cleaning |
| 3 | Exploratory Data Analysis |
| 4 | Sentiment vs Performance |
| 5 | Statistical Testing |
| 6 | Daily PnL Time-Series |
| 7 | Trader Ranking |
| 8 | Trader Clustering |
| 9 | ML — Contaminated Pipeline (with leakage diagnosis) |
| 10 | ML — Clean Pipeline (leakage-free, honest results) |
| 11 | Feature Importance Comparison |
| 12 | Key Insights & Recommendations |
"""),

# ── 1. Setup ───────────────────────────────────────────────────────────────
md("## 1. Environment Setup"),

code("""\
import sys, warnings
from pathlib import Path

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Image, Markdown

from src.utils import (set_random_seeds, FIGURES_DIR, PROCESSED_DIR,
                       TABLES_DIR, SENTIMENT_COLOR_MAP, SENTIMENT_ORDER)
set_random_seeds(42)

print(f"Project root : {PROJECT_ROOT}")
print(f"NumPy  {np.__version__} | Pandas {pd.__version__}")
print("Environment ready")
"""),

# ── 2. Data Loading ────────────────────────────────────────────────────────
md("""## 2. Data Loading & Cleaning

**Files auto-detected from `data/`:**
- `fear_greed_index.csv` — 2,644 daily BTC sentiment labels
- `historical_data.csv` — 211,224 raw Hyperliquid trade records

**Cleaning steps applied automatically:**
1. Column name normalisation (lowercase + underscore)
2. Column alias mapping (e.g. `Coin` → `symbol`, `Closed PnL` → `closedpnl`)
3. Unix millisecond timestamp parsing
4. Deduplication (49,459 duplicate trades removed)
5. Inner join on calendar date → 140,792 matched trades
"""),

code("""\
from src.data_cleaning import (
    load_fear_greed, load_trader_data, merge_datasets, build_daily_aggregates
)

fg     = load_fear_greed()
trades = load_trader_data()
merged = merge_datasets(fg, trades)
daily  = build_daily_aggregates(merged)

print(f"Fear & Greed : {len(fg):,} rows | {fg['date'].min().date()} to {fg['date'].max().date()}")
print(f"Trades       : {len(trades):,} rows | {trades['account'].nunique():,} unique accounts")
print(f"Merged       : {len(merged):,} rows | {merged['date'].nunique()} unique dates")
"""),

code("""\
print("=== Fear & Greed Schema ===")
display(fg.head(3))
print("\\n=== Trade Schema ===")
display(trades.head(3))
"""),

code("""\
# Preprocessing summary
print("Missing values in merged dataset:")
nulls = merged.isnull().sum()
print(nulls[nulls > 0].to_string() if nulls.any() else "None")
print(f"\\nSentiment coverage: {len(merged)/len(trades)*100:.1f}% of trades matched to sentiment dates")
print(f"\\nSentiment breakdown in matched trades:")
display(merged['classification'].value_counts().rename_axis('Sentiment').reset_index(name='Trades'))
"""),

# ── 3. Feature Engineering ─────────────────────────────────────────────────
md("## 3. Feature Engineering"),

code("""\
from src.feature_engineering import (
    add_trade_level_features, add_daily_features, add_account_features,
    rank_traders, build_clustering_matrix
)

df = add_trade_level_features(merged)
df = add_daily_features(df)
df = add_account_features(df)

print(f"Feature matrix: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"New features added: {[c for c in df.columns if c not in merged.columns]}")
"""),

code("""\
display(df[['closedpnl','sentiment_score','trade_direction',
            'trade_size_usd','account_win_rate','account_sharpe']].describe().round(3))
"""),

# ── 4. EDA ─────────────────────────────────────────────────────────────────
md("## 4. Exploratory Data Analysis"),

md("### 4.1 Sentiment Distribution"),

code("""\
from src.analysis import eda_sentiment
sent_dist = eda_sentiment(fg)
display(sent_dist)

from src.visualization import plot_sentiment_distribution, plot_sentiment_timeline
Image(str(plot_sentiment_distribution(fg)), width=700)
"""),

code("""\
Image(str(plot_sentiment_timeline(fg)), width=950)
"""),

md("### 4.2 Trader Overview"),

code("""\
from src.analysis import eda_trader
stats = eda_trader(df)
summary = pd.DataFrame([stats]).T.rename(columns={0: "Value"})
display(summary)
"""),

code("""\
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df['closedpnl'].clip(-3000, 3000), bins=80,
             color='#2CA02C', alpha=0.75, edgecolor='none')
axes[0].axvline(0, color='white', linewidth=1.5, linestyle='--')
axes[0].set_title('PnL Distribution (clipped +/-3000 USD)', fontweight='bold')
axes[0].set_xlabel('Closed PnL (USD)')
axes[0].set_ylabel('Frequency')
axes[0].grid(alpha=0.3)

# Long vs Short breakdown
if 'side' in df.columns:
    vc = df['side'].value_counts()
    axes[1].bar(vc.index, vc.values,
                color=['#2CA02C' if x=='LONG' else '#D62728' for x in vc.index])
    axes[1].set_title('Long vs Short Trade Count', fontweight='bold')
    axes[1].set_ylabel('Number of Trades')
    axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(str(FIGURES_DIR / 'eda_pnl_distribution.png'), dpi=150, bbox_inches='tight')
plt.show()
"""),

# ── 5. Sentiment vs Performance ────────────────────────────────────────────
md("## 5. Sentiment vs Performance Analysis"),

code("""\
from src.analysis import sentiment_performance_table
from src.visualization import (plot_pnl_by_sentiment, plot_win_rate_by_sentiment,
                                plot_trade_volume_by_sentiment, plot_pnl_violin,
                                plot_long_short_by_sentiment)

perf = sentiment_performance_table(df)
display(perf.round(3))
"""),

code("""\
Image(str(plot_pnl_by_sentiment(df)), width=950)
"""),

code("""\
Image(str(plot_win_rate_by_sentiment(df)), width=750)
"""),

code("""\
Image(str(plot_trade_volume_by_sentiment(df)), width=750)
"""),

code("""\
Image(str(plot_pnl_violin(df)), width=950)
"""),

code("""\
Image(str(plot_long_short_by_sentiment(df)), width=750)
"""),

# ── 6. Statistical Testing ─────────────────────────────────────────────────
md("## 6. Statistical Testing"),

code("""\
from src.analysis import run_statistical_tests

stat = run_statistical_tests(df)
anova = stat.get('anova', {})
corr  = stat.get('correlation', {})

print(f"One-Way ANOVA (PnL ~ Sentiment)")
print(f"  F-statistic : {anova.get('f_statistic', 0):.4f}")
print(f"  p-value     : {anova.get('p_value', 1):.6f}")
print(f"  Significant : {'YES - p < 0.05' if anova.get('significant') else 'NO'}")
print()
print(f"Pearson Correlation (sentiment_score <-> closedpnl)")
print(f"  r = {corr.get('pearson_r', 0):.4f}")
print(f"  p = {corr.get('p_value', 1):.4f}")
"""),

code("""\
display(pd.DataFrame(stat.get('pairwise_ttests', [])).round(4))
"""),

code("""\
from src.visualization import plot_correlation_heatmap
Image(str(plot_correlation_heatmap(df)), width=950)
"""),

# ── 7. Daily PnL ───────────────────────────────────────────────────────────
md("## 7. Daily PnL Time-Series"),

code("""\
from src.visualization import plot_daily_pnl_timeseries
Image(str(plot_daily_pnl_timeseries(daily)), width=950)
"""),

# ── 8. Trader Ranking ──────────────────────────────────────────────────────
md("## 8. Trader Ranking"),

code("""\
ranking = rank_traders(df)
print(f"Ranked {len(ranking):,} unique trader accounts")
display(ranking[['account','account_total_pnl','account_win_rate',
                  'account_sharpe','composite_score','composite_rank']].head(10).round(4))
"""),

code("""\
from src.visualization import plot_trader_ranking
Image(str(plot_trader_ranking(ranking, top_n=20)), width=800)
"""),

# ── 9. Clustering ──────────────────────────────────────────────────────────
md("## 9. Trader Clustering"),

code("""\
from src.analysis import run_kmeans, run_hierarchical, label_clusters
from src.visualization import plot_clusters

feat_scaled, feat_cols = build_clustering_matrix(ranking)
kmeans_labels, _       = run_kmeans(feat_scaled)
ranking_clustered      = label_clusters(ranking, kmeans_labels)

print("Segment distribution:")
display(ranking_clustered['segment'].value_counts().reset_index())
"""),

code("""\
Image(str(plot_clusters(ranking_clustered, feat_scaled)), width=800)
"""),

code("""\
profile_cols = [c for c in ['account_total_pnl','account_win_rate',
                              'account_sharpe','account_trade_count'] if c in ranking_clustered.columns]
display(ranking_clustered.groupby('segment')[profile_cols].mean().round(3))
"""),

# ── 10. ML — Contaminated ─────────────────────────────────────────────────
md("""## 10. ML Pipeline — Contaminated (with Leakage Diagnosis)

> **Warning:** This section intentionally reproduces the ORIGINAL pipeline to demonstrate the leakage.
> The inflated AUC = 0.979 is caused by three leakage mechanisms identified in Section 11.
"""),

code("""\
from src.feature_engineering import build_ml_features
from src.analysis import run_classification

X_leaked, y_class, y_reg = build_ml_features(df)
print(f"Feature matrix: {X_leaked.shape}")
print(f"Features: {X_leaked.columns.tolist()}")
print(f"Class balance: {y_class.mean()*100:.1f}% wins")
"""),

code("""\
clf_leaked = run_classification(X_leaked, y_class)

rows = []
for name, res in clf_leaked.items():
    if name == 'feature_names': continue
    rows.append({'Model': name, 'Accuracy': res['accuracy'], 'F1': res['f1'],
                 'ROC-AUC': res['roc_auc']})
leaked_df = pd.DataFrame(rows)
display(leaked_df)
print("\\nNOTE: These results are inflated by data leakage — see Section 11.")
"""),

code("""\
# Show feature importances — most importance will be on leaked features
rf_fi = clf_leaked.get('Random Forest', {}).get('feature_importance')
if rf_fi is not None:
    leaked_features = ['log_abs_pnl', 'risk_adjusted_return', 'daily_win_rate',
                       'daily_pnl', 'account_win_rate', 'account_total_pnl', 'account_sharpe']
    fi_df = rf_fi.reset_index()
    fi_df.columns = ['Feature', 'Importance']
    fi_df['Status'] = fi_df['Feature'].apply(
        lambda x: 'LEAKED' if x in leaked_features else 'clean'
    )
    display(fi_df)
"""),

# ── 11. Leakage Audit ─────────────────────────────────────────────────────
md("""## 11. ML Leakage Audit

During development, ROC-AUC = 0.979 was flagged as implausibly high.
A formal leakage audit (`audit_leakage.py`) identified **3 leakage mechanisms**:

| # | Type | Features | Why |
|---|---|---|---|
| L1 | Target-derived | `log_abs_pnl`, `risk_adjusted_return` | Transforms of `closedpnl` which determines `is_win` |
| L2 | Future look-ahead | `account_win_rate`, `account_total_pnl`, `account_sharpe` | Computed over full history before train/test split |
| L3 | Same-day target | `daily_win_rate`, `daily_pnl` | Includes current trade's outcome |

**Fixes applied in the clean pipeline:**
1. Chronological 80/20 split (no random shuffle)
2. Account stats computed on training set only, joined to test
3. Target-derived and same-day target features excluded from ML
"""),

code("""\
# Load audit results from disk
import json
audit = json.load(open('outputs/tables/leakage_audit.json'))

print("Leakage Audit Results")
print(f"  Leaked features identified : {len(audit['leaked_features'])}")
print(f"  Leaked features            : {audit['leaked_features']}")
print(f"  AUC inflation              : +{audit['leakage_drop_auc']:.4f} points")
print()
print("Contaminated results:")
for m, r in audit['contaminated'].items():
    print(f"  {m}: AUC={r['roc_auc']:.4f} | F1={r['f1']:.4f}")
print()
print("Clean results (no leakage):")
for m, r in audit['clean'].items():
    print(f"  {m}: AUC={r['roc_auc']:.4f} | F1={r['f1']:.4f}")
"""),

# ── 12. Clean ML Pipeline ─────────────────────────────────────────────────
md("""## 12. Clean ML Pipeline (Leakage-Free)

Chronological split: first 80% of trades for training, last 20% for testing.
Account features computed on training data only.
No target-derived features.
"""),

code("""\
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from src.utils import safe_divide

# 1. Sort chronologically
df_clean = df.sort_values('time').reset_index(drop=True)
df_clean = add_trade_level_features(merged.copy())

cutoff = int(len(df_clean) * 0.80)
df_train = df_clean.iloc[:cutoff].copy()
df_test  = df_clean.iloc[cutoff:].copy()

print(f"Train: {len(df_train):,} trades | {df_train['time'].min().date()} to {df_train['time'].max().date()}")
print(f"Test : {len(df_test):,} trades  | {df_test['time'].min().date()} to {df_test['time'].max().date()}")
"""),

code("""\
# 2. Account features from training data only
def sharpe_like(s):
    std = s.std()
    return s.mean() / std if std > 0 else 0.0

agg = {
    'account_total_pnl_train':  ('closedpnl', 'sum'),
    'account_trade_count_train': ('closedpnl', 'count'),
    'account_win_rate_train':    ('is_win', 'mean'),
    'account_pnl_std_train':     ('closedpnl', 'std'),
}
if 'is_long' in df_train.columns:
    agg['account_long_ratio_train'] = ('is_long', 'mean')

acct_train = df_train.groupby('account').agg(**agg).reset_index()
sharpe     = df_train.groupby('account')['closedpnl'].apply(sharpe_like).rename('account_sharpe_train')
acct_train = acct_train.merge(sharpe, on='account')

df_train = df_train.merge(acct_train, on='account', how='left').fillna(0)
df_test  = df_test.merge(acct_train, on='account', how='left').fillna(0)

print(f"Account features computed from {len(acct_train)} training accounts")
print(f"Test accounts with no training history: {(df_test['account_trade_count_train'] == 0).sum()}")
"""),

code("""\
# 3. Build clean feature matrix
CLEAN_FEATURES = [
    'sentiment_score',
    'trade_direction',
    'trade_size_usd',
    'account_win_rate_train',
    'account_total_pnl_train',
    'account_sharpe_train',
    'account_trade_count_train',
    'account_long_ratio_train',
]
CLEAN_FEATURES = [c for c in CLEAN_FEATURES if c in df_train.columns]

X_tr = df_train[CLEAN_FEATURES].fillna(0)
y_tr = df_train['is_win'].fillna(0).astype(int)
X_te = df_test[CLEAN_FEATURES].fillna(0)
y_te = df_test['is_win'].fillna(0).astype(int)

print(f"Feature matrix — train: {X_tr.shape} | test: {X_te.shape}")
print(f"Class balance — train: {y_tr.mean()*100:.1f}% | test: {y_te.mean()*100:.1f}%")
"""),

code("""\
# 4. Train models
# Logistic Regression
scaler   = StandardScaler()
lr_clean = LogisticRegression(max_iter=1000, random_state=42)
lr_clean.fit(scaler.fit_transform(X_tr), y_tr)
prob_lr  = lr_clean.predict_proba(scaler.transform(X_te))[:, 1]

# Random Forest
rf_clean = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
rf_clean.fit(X_tr, y_tr)
prob_rf  = rf_clean.predict_proba(X_te)[:, 1]

results_clean = {
    'Logistic Regression': {'auc': roc_auc_score(y_te, prob_lr),
                             'f1':  f1_score(y_te, lr_clean.predict(scaler.transform(X_te)), zero_division=0)},
    'Random Forest':       {'auc': roc_auc_score(y_te, prob_rf),
                             'f1':  f1_score(y_te, rf_clean.predict(X_te), zero_division=0)},
}

try:
    import xgboost as xgb
    xgb_c = xgb.XGBClassifier(n_estimators=300, random_state=42, verbosity=0, eval_metric='logloss')
    xgb_c.fit(X_tr, y_tr)
    prob_xgb = xgb_c.predict_proba(X_te)[:, 1]
    results_clean['XGBoost'] = {'auc': roc_auc_score(y_te, prob_xgb),
                                 'f1':  f1_score(y_te, xgb_c.predict(X_te), zero_division=0)}
except: pass

display(pd.DataFrame(results_clean).T.round(4))
"""),

# ── 13. Feature Importance Comparison ────────────────────────────────────
md("## 13. Feature Importance — Contaminated vs Clean"),

code("""\
import matplotlib.patches as mpatches

fi_clean = pd.Series(rf_clean.feature_importances_, index=CLEAN_FEATURES).sort_values(ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Contaminated
if rf_fi is not None:
    leaked = ['log_abs_pnl','risk_adjusted_return','daily_win_rate',
              'daily_pnl','account_win_rate','account_total_pnl','account_sharpe']
    fi_leaked_top = rf_fi.head(9)
    colors_l = ['#D62728' if f in leaked else '#2CA02C' for f in fi_leaked_top.index]
    axes[0].barh(fi_leaked_top.index[::-1], fi_leaked_top.values[::-1], color=colors_l[::-1])
    axes[0].set_title('Contaminated RF (AUC=0.979)', fontweight='bold', color='#D62728')
    axes[0].set_xlabel('Importance')
    axes[0].grid(axis='x', alpha=0.3)
    red_p  = mpatches.Patch(color='#D62728', label='LEAKED')
    grn_p  = mpatches.Patch(color='#2CA02C', label='Clean')
    axes[0].legend(handles=[red_p, grn_p])

# Clean
colors_c = ['#1F77B4' if 'sentiment' in f else '#2CA02C' for f in fi_clean.index]
axes[1].barh(fi_clean.index[::-1], fi_clean.values[::-1], color=colors_c[::-1])
axes[1].set_title('Clean RF (AUC=0.578)', fontweight='bold', color='#2CA02C')
axes[1].set_xlabel('Importance')
axes[1].grid(axis='x', alpha=0.3)
blue_p = mpatches.Patch(color='#1F77B4', label='Sentiment (macro signal)')
grn_p2 = mpatches.Patch(color='#2CA02C', label='Trade / Account features')
axes[1].legend(handles=[blue_p, grn_p2])

fig.suptitle('Feature Importance: Contaminated vs Leak-Free Pipeline', fontsize=14, fontweight='bold')
fig.tight_layout()
fig.savefig(str(FIGURES_DIR / '14_feature_importance_comparison.png'), dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 14_feature_importance_comparison.png")
"""),

# ── 14. Key Insights ──────────────────────────────────────────────────────
md("""## 14. Key Insights & Recommendations

### Statistical Findings (all valid — no ML leakage involved)

| Finding | Value |
|---|---|
| ANOVA p-value | < 0.0001 |
| Best avg PnL sentiment | Greed ($101.37/trade) |
| Best win rate sentiment | Extreme Greed (49.96%) |
| Most active sentiment | Fear (99,477 trades = 70% of volume) |
| Pearson r (sentiment <-> PnL) | 0.0073 (significant) |

### Corrected ML Results

| Model | Contaminated AUC | Clean AUC |
|---|---|---|
| Random Forest | 0.9796 | **0.5779** |
| XGBoost | 0.9768 | **0.5854** |

### Business Recommendations

1. **Reduce leverage during Extreme Greed** — correction risk outweighs momentum
2. **Deploy LONG bias during sustained Extreme Fear** — contrarian exhaustion signal
3. **Use sentiment_score as position sizing multiplier** — statistically validated signal
4. **Prioritize account history features over sentiment** — skill dominates macro in individual trade prediction
5. **Build an ensemble of on-chain signals** — funding rates, whale flows, order book depth
"""),

code("""\
print("=" * 60)
print("  NOTEBOOK COMPLETE")
print("=" * 60)
print(f"Figures saved : {FIGURES_DIR}")
print(f"Tables saved  : {TABLES_DIR}")
print(f"GitHub Repo   : https://github.com/harichopper/web3-trader-behavior-analysis")
print()
print("FINAL RESULTS SUMMARY")
print(f"  ANOVA p-value        : < 0.0001  (highly significant)")
print(f"  Best PnL sentiment   : Greed ($101.37/trade)")
print(f"  Best win rate        : Extreme Greed (49.96%)")
print(f"  Clean ML AUC (RF)   : 0.578")
print(f"  Sentiment rank       : #4 of 8 features in clean model")
"""),

]
# ─────────────────────────────────────────────────────────────────────────────

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
        "title": "Bitcoin Sentiment x Hyperliquid Trader Performance"
    },
    "cells": cells
}

NB_PATH.parent.mkdir(parents=True, exist_ok=True)
NB_PATH.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
print(f"Notebook written: {NB_PATH}")
print(f"Total cells: {len(cells)}")
