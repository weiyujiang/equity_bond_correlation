# Equity–Bond Correlation & Macro Regime Analysis

A self-contained Python pipeline that studies **why the correlation between Canadian equities and bonds changes over time** — and what it means for multi-asset portfolio construction.

The core idea: equity–bond correlation is not a fixed number. It flips sign depending on the dominant macroeconomic shock. During demand-driven expansions, bonds hedge equity risk (negative correlation). During inflation or supply shocks — like 2021–2022 — both assets fall together (positive correlation), breaking the traditional 60/40 logic.

---

## Research Question

> *Under which macroeconomic regimes does the equity–bond diversification relationship break down, and how should multi-asset portfolios adapt?*

---

## Methodology

The pipeline combines three complementary approaches:

### 1. Sign-Restricted Structural VAR (SVAR)

A reduced-form VAR is estimated on three monthly Canadian macro variables:

| Variable | Source | Transformation |
|---|---|---|
| GDP growth | Statistics Canada (Table 36-10-0434-01) | Log-differenced, ×100 |
| YoY inflation | Statistics Canada (Table 18-10-0004-01) | CPI % change vs. year-ago |
| Policy rate | Bank of Canada Valet API (V39079) | Level (overnight rate target) |

The structural shocks are identified via **sign restrictions** on the contemporaneous impulse matrix. The algorithm draws random orthonormal rotations (Cholesky-based) until it finds one consistent with:

| Shock | GDP | Inflation | Policy Rate |
|---|---|---|---|
| Demand | + | + | + |
| Supply / Inflation | − | + | + |
| Monetary Policy | − | − | + |

Each month is then labeled by its **dominant shock** — the structural disturbance with the largest absolute magnitude that month. A **robustness specification** replaces inflation and policy rate levels with their first differences to reduce persistence.

### 2. Markov-Switching Regime Models

Two complementary hidden-state models capture persistent regime shifts:

- **Inflation regimes**: A 2-state Markov-switching regression on YoY inflation identifies *high-inflation* vs. *low-inflation* periods (e.g., the 2021–2023 episode vs. the 2010–2019 low-rate era).
- **Supply-shock regimes**: A 2-state Markov-switching regression on the SVAR supply-shock series identifies months where supply disruptions are *large and persistent* vs. *normal*. This answers a distinct question from the inflation model — whether supply shocks are in an elevated-volatility state, regardless of whether CPI has fully responded.

Both models use switching variance to allow each state to have its own shock volatility, and are estimated via EM with multiple random restarts.

### 3. DCC-GARCH (Dynamic Conditional Correlation)

A two-asset DCC-GARCH model produces the time-varying equity–bond correlation directly from returns, without imposing macro regime labels:

- **Step 1**: Individual GARCH(1,1) models for XIC (Canadian equities) and XBB (Canadian bonds), estimated by maximum likelihood via Nelder-Mead / BFGS.
- **Step 2**: DCC parameters (α, β) are estimated on the GARCH-standardized residuals.
- **Output**: A monthly `dcc_corr` series showing how the equity–bond correlation has evolved since 2002.

The DCC correlation is then cross-tabulated against both SVAR regimes and Markov regimes to validate that the structural shock labels correspond to genuine changes in the statistical correlation.

---

## Data Sources

All data is pulled automatically at runtime from public APIs — no manual downloads required.

| Series | Source | API |
|---|---|---|
| XIC.TO (Canadian equities ETF) | Yahoo Finance | Chart endpoint |
| XBB.TO (Canadian broad bond ETF) | Yahoo Finance | Chart endpoint |
| XSB.TO (short-term bond ETF) | Yahoo Finance | Chart endpoint |
| CGL-C.TO / CGL.TO (gold, hedged/unhedged) | Yahoo Finance | Chart endpoint |
| DBC (broad commodities) | Yahoo Finance | Chart endpoint |
| FXC (CAD/USD proxy) | Yahoo Finance | Chart endpoint |
| CPI (All-items, Canada) | Statistics Canada | Table 18-10-0004-01 |
| Real GDP (seasonally adjusted) | Statistics Canada | Table 36-10-0434-01 |
| Overnight rate target | Bank of Canada Valet | Series V39079 |

Sample period: **March 2002 – present** (limited by ETF inception dates).

---

## Portfolio Construction

Beyond correlation analysis, the pipeline evaluates 27 multi-asset portfolio variants under each macro regime. Portfolios combine:

- **Core**: XIC (equities) + XBB (broad bonds)
- **Alternatives**: XSB (short bonds), gold hedged/unhedged (CGL.TO / CGL-C.TO), commodities (DBC), CAD/USD (FXC)

Weight variants range from a simple **60/40** baseline to real-asset-augmented mixes (e.g., 50% equity / 25% broad bonds / 15% short bonds / 10% gold). For each portfolio × regime combination the pipeline reports:

- Annualized return and volatility
- Maximum drawdown
- Historical 5% VaR
- Sharpe ratio (using XSB as the risk-free proxy)
- Difference vs. 60/40 baseline

This answers the practical question: *which portfolio structure holds up best when supply shocks make bonds a poor hedge?*

---

## Outputs

Running `bond_equity_svar_pipeline.py` writes all results to `outputs/`:

| File | Description |
|---|---|
| `xic_xbb_monthly_returns.csv` | Raw monthly returns for XIC and XBB |
| `correlation_by_period.csv` | Rolling-window equity–bond correlation by historical epoch |
| `macro_monthly_transformed.csv` | GDP growth, inflation, and policy rate used in the VAR |
| `svar_structural_shocks.csv` | Monthly demand / supply / monetary shock magnitudes |
| `svar_dominant_regime.csv` | Month-by-month dominant shock label |
| `svar_signed_regime.csv` | Signed regime (e.g., `supply_inflation_positive`) |
| `correlation_by_svar_regime.csv` | Equity–bond correlation conditional on SVAR regime |
| `portfolio_60_40_by_svar_regime.csv` | 60/40 performance by SVAR regime |
| `baseline_2021_2022_regime_timeline.csv` | Regime-by-regime breakdown of the 2021–2022 shock |
| `markov_supply_shock_regimes.csv` | Markov smoothed/filtered probabilities for supply regimes |
| `dcc_garch_*.csv` | DCC-GARCH correlation series and regime summaries |
| `portfolio_asset_monthly_returns.csv` | Monthly returns for all 7 portfolio assets |
| `portfolio_monthly_returns.csv` | Monthly returns for all 27 portfolio variants |
| `portfolio_performance_by_supply_regime.csv` | Full performance table by regime for all portfolios |
| `robustness_*.csv` | All of the above repeated under the changes-in-levels SVAR |

---

## Usage

```bash
pip install numpy pandas scipy statsmodels requests
python bond_equity_svar_pipeline.py
```

The script is intentionally linear (notebook-style): run top to bottom, read the printed checkpoint tables, then use the CSV outputs for further analysis or charting.

---

## Key Findings

- **Equity–bond correlation is regime-dependent.** In demand-shock months (growth-driven expansions), the correlation is negative — bonds hedge equity drawdowns as expected. In supply/inflation-shock months, both assets decline together and correlation turns positive, undermining diversification.
- **The 2021–2022 episode** is classified overwhelmingly as a high-pressure supply shock: the SVAR, Markov inflation model, and DCC-GARCH all agree that correlation rose sharply, consistent with the historical parallel to the 1970s stagflation period.
- **Gold and short bonds partially offset the correlation breakdown** during high-pressure supply regimes. Portfolios augmenting the 60/40 baseline with real assets show lower drawdowns in those periods, at a modest cost in normal-regime performance.
- **Monetary policy shocks** produce a distinct pattern: GDP and inflation both fall while rates rise, and equity–bond correlation is ambiguous — hedging effectiveness depends on whether the tightening cycle is front-loaded.

---

## Related Work

This pipeline supports the broader research program on how macroeconomic structure affects asset pricing and portfolio construction:

- [Foreign Investors' Industry Expertise and Firm Value](https://github.com/weiyujiang/research-portfolio/blob/main/WeiYuJiang_JMP.pdf) — how macro-industry expertise of foreign investors affects firm value across 70 countries.
- [Foreign Ownership Diversity and Board Diversity](https://github.com/weiyujiang/research-portfolio/blob/main/WeiYuJiang_WorkingPaper.pdf) — governance channel through which foreign investors influence firm outcomes.
- [Credit Value Adjustment with Market-implied Recovery](https://github.com/weiyujiang/research-portfolio/blob/main/WeiYuJiang_PublishedPaper.pdf) — multi-asset credit risk modeling under regime-varying recovery rates.
