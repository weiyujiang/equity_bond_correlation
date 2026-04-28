"""
Bond-equity correlation regime project: public-data SVAR pipeline.

Purpose
-------
1. Pull public monthly data for Canadian equities, Canadian bonds, GDP, CPI,
   and policy/rate proxies.
2. Estimate a compact VAR on macro variables.
3. Classify each month by the dominant sign-restricted shock:
   demand, supply/inflation, or monetary policy.
4. Merge shock labels with XIC/XBB monthly returns.
5. Output regime-conditioned bond-equity correlation and 60/40 statistics.

Notes
-----
This file is intentionally notebook-like: run it top to bottom, inspect the
printed checkpoint tables, then use the CSV outputs in the research brief.

Some public data endpoints occasionally change column names. The StatCan
selector helpers below print candidates if an exact filter fails.
"""

from __future__ import annotations

import io
import math
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.optimize import minimize
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from statsmodels.tsa.api import VAR


START = "2002-03-01"
END = None  # None = latest available
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Data fetch helpers
# ---------------------------------------------------------------------------


def yahoo_monthly_adjclose(symbol: str, start: str = START, end: str | None = END) -> pd.Series:
    """Fetch Yahoo monthly adjusted prices through the chart endpoint."""
    start_ts = int(pd.Timestamp(start, tz="UTC").timestamp())
    end_ts = int((pd.Timestamp(end, tz="UTC") if end else pd.Timestamp.utcnow()).timestamp())
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?period1={start_ts}&period2={end_ts}&interval=1mo"
        "&events=history%7Cdiv%7Csplit&includeAdjustedClose=true"
    )
    r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    payload = r.json()
    result = payload["chart"]["result"][0]
    timestamps = result["timestamp"]
    adj = result["indicators"].get("adjclose", [{}])[0].get("adjclose")
    if adj is None:
        adj = result["indicators"]["quote"][0]["close"]

    idx = pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(None).to_period("M").to_timestamp("M")
    s = pd.Series(adj, index=idx, name=symbol).dropna().astype(float)
    return s[~s.index.duplicated(keep="last")]


def to_month_end_index(s: pd.Series) -> pd.Series:
    """Normalize a monthly/daily series to month-end timestamps."""
    x = s.copy()
    x.index = pd.to_datetime(x.index).to_period("M").to_timestamp("M")
    x = x.sort_index()
    return x[~x.index.duplicated(keep="last")]


def statcan_table(table_id: str) -> pd.DataFrame:
    """
    Download an entire StatCan table zip.

    Example:
    - CPI table 18-10-0004-01 -> 18100004
    - GDP table 36-10-0434-01 -> 36100434
    """
    url = f"https://www150.statcan.gc.ca/n1/tbl/csv/{table_id}-eng.zip"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        csv_name = [n for n in zf.namelist() if n.endswith(".csv") and "MetaData" not in n][0]
        with zf.open(csv_name) as f:
            return pd.read_csv(f)


def boc_valet_series(series_name: str, start: str = START, end: str | None = END) -> pd.Series:
    """Fetch a Bank of Canada Valet API series."""
    url = f"https://www.bankofcanada.ca/valet/observations/{series_name}/json"
    params = {"start_date": pd.Timestamp(start).strftime("%Y-%m-%d")}
    if end:
        params["end_date"] = pd.Timestamp(end).strftime("%Y-%m-%d")
    r = requests.get(url, params=params, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    observations = r.json()["observations"]
    rows = []
    for obs in observations:
        value = obs.get(series_name, {}).get("v")
        if value not in (None, ""):
            rows.append((pd.Timestamp(obs["d"]), float(value)))
    s = pd.Series(dict(rows)).sort_index()
    s.name = series_name
    return s


def show_candidates(df: pd.DataFrame, column_pattern: str, value_pattern: str, limit: int = 25) -> None:
    col = next(c for c in df.columns if re.search(column_pattern, c, re.I))
    vals = sorted(str(x) for x in df[col].dropna().unique() if re.search(value_pattern, str(x), re.I))
    print(f"\nCandidates in column '{col}' matching '{value_pattern}':")
    for v in vals[:limit]:
        print("  -", v)


def pick_series(
    df: pd.DataFrame,
    *,
    value_name: str,
    filters: dict[str, str],
    date_col: str = "REF_DATE",
) -> pd.Series:
    """Filter a StatCan-style table using regex filters on column names and values."""
    x = df.copy()
    for col_pat, val_pat in filters.items():
        matches = [c for c in x.columns if re.search(col_pat, c, re.I)]
        if not matches:
            raise KeyError(f"No column matching {col_pat}. Columns: {list(x.columns)}")
        col = matches[0]
        mask = x[col].astype(str).str.contains(val_pat, case=False, regex=True, na=False)
        x = x[mask]
        if x.empty:
            show_candidates(df, col_pat, val_pat)
            raise ValueError(f"No rows left after filter {col} ~= {val_pat}")

    x[date_col] = pd.to_datetime(x[date_col])
    s = x.set_index(date_col)["VALUE"].astype(float).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    s.name = value_name
    return to_month_end_index(s)


# ---------------------------------------------------------------------------
# Transformations and statistics
# ---------------------------------------------------------------------------


def monthly_returns(price: pd.Series) -> pd.Series:
    r = price.pct_change().dropna()
    r.name = price.name
    return r


def max_drawdown(r: pd.Series) -> float:
    wealth = (1 + r).cumprod()
    return (wealth / wealth.cummax() - 1).min()


def perf_stats(r: pd.Series) -> dict[str, float]:
    return {
        "obs": len(r),
        "ann_return": (1 + r).prod() ** (12 / len(r)) - 1,
        "ann_vol": r.std(ddof=1) * math.sqrt(12),
        "best_month": r.max(),
        "worst_month": r.min(),
        "max_drawdown": max_drawdown(r),
        "skew": r.skew(),
        "hist_var_5": r.quantile(0.05),
    }


def corr_by_period(returns: pd.DataFrame) -> pd.DataFrame:
    periods = {
        "Full sample": ("2002-03-01", None),
        "2002-2007": ("2002-03-01", "2007-12-31"),
        "2008-2009": ("2008-01-01", "2009-12-31"),
        "2010-2020": ("2010-01-01", "2020-12-31"),
        "2021-present": ("2021-01-01", None),
    }
    rows = []
    for label, (start, end) in periods.items():
        x = returns.loc[start:end] if end else returns.loc[start:]
        rows.append({"period": label, "obs": len(x), "corr": x.iloc[:, 0].corr(x.iloc[:, 1])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# SVAR approximation with sign-restricted rotations
# ---------------------------------------------------------------------------


@dataclass
class ShockSpec:
    name: str
    signs: tuple[int, int, int]  # GDP, inflation, policy rate impulse signs


SHOCKS = [
    ShockSpec("demand", (+1, +1, +1)),
    ShockSpec("supply_inflation", (-1, +1, +1)),
    ShockSpec("monetary_policy", (-1, -1, +1)),
]


def random_orthonormal(n: int, rng: np.random.Generator) -> np.ndarray:
    z = rng.normal(size=(n, n))
    q, r = np.linalg.qr(z)
    d = np.sign(np.diag(r))
    return q * d


def sign_restricted_structural_shocks(
    macro: pd.DataFrame,
    lags: int = 6,
    draws: int = 5000,
    seed: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Estimate reduced-form VAR and find one rotation whose contemporaneous impulse
    columns match demand, supply/inflation, and policy sign restrictions.

    This is a pragmatic sign-restricted SVAR implementation for the research
    checkpoint. For publication-quality inference, keep multiple accepted draws
    and report uncertainty bands.
    """
    model = VAR(macro.dropna())
    res = model.fit(lags)
    resid = res.resid
    sigma = np.asarray(res.sigma_u)
    chol = np.linalg.cholesky(sigma)
    rng = np.random.default_rng(seed)

    wanted = np.array([s.signs for s in SHOCKS]).T
    accepted_a = None
    for _ in range(draws):
        q = random_orthonormal(3, rng)
        a = chol @ q
        signs = np.sign(a)
        if np.array_equal(signs, wanted):
            accepted_a = a
            break
        # Column signs are arbitrary; try flipping columns.
        for flips in np.array(np.meshgrid(*[[-1, 1]] * 3)).T.reshape(-1, 3):
            af = a * flips
            if np.array_equal(np.sign(af), wanted):
                accepted_a = af
                break
        if accepted_a is not None:
            break

    if accepted_a is None:
        raise RuntimeError("No accepted sign-restricted rotation found. Increase draws or relax signs.")

    structural = pd.DataFrame(
        np.linalg.solve(accepted_a, resid.T).T,
        index=resid.index,
        columns=[s.name for s in SHOCKS],
    )
    dominant = structural.abs().idxmax(axis=1).rename("dominant_shock").to_frame()
    return structural, dominant


def signed_regime_labels(structural: pd.DataFrame) -> pd.DataFrame:
    """Classify each month by dominant shock type and sign."""
    dominant = structural.abs().idxmax(axis=1)
    values = np.array([structural.at[idx, col] for idx, col in zip(structural.index, dominant)])
    labels = [
        f"{shock}_{'positive' if value >= 0 else 'negative'}"
        for shock, value in zip(dominant, values)
    ]
    return pd.DataFrame(
        {
            "dominant_shock": dominant,
            "dominant_shock_value": values,
            "signed_regime": labels,
        },
        index=structural.index,
    )


def regime_portfolio_tables(returns: pd.DataFrame, regimes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = returns.join(regimes, how="inner")
    x["portfolio_60_40"] = 0.6 * x.iloc[:, 0] + 0.4 * x.iloc[:, 1]

    corr_rows = []
    perf_rows = []
    for regime, g in x.groupby("dominant_shock"):
        corr_rows.append(
            {
                "regime": regime,
                "obs": len(g),
                "xic_xbb_corr": g.iloc[:, 0].corr(g.iloc[:, 1]),
            }
        )
        stats = perf_stats(g["portfolio_60_40"])
        perf_rows.append({"regime": regime, **stats})
    return pd.DataFrame(corr_rows), pd.DataFrame(perf_rows)


def validation_tables(
    macro: pd.DataFrame,
    returns: pd.DataFrame,
    structural: pd.DataFrame,
    regimes: pd.DataFrame,
    prefix: str,
) -> None:
    """Write and print economic validation tables for SVAR labels."""
    labels = signed_regime_labels(structural)
    data = macro.join(returns, how="inner").join(labels, how="inner")

    macro_asset = (
        data.groupby("dominant_shock")
        .agg(
            months=("dominant_shock", "size"),
            gdp_growth_mean=("gdp_growth", "mean"),
            inflation_mean=("inflation_yoy", "mean"),
            policy_rate_mean=("policy_rate", "mean"),
            xic_monthly_mean=("xic", "mean"),
            xbb_monthly_mean=("xbb", "mean"),
            xic_xbb_corr=("xic", lambda s: s.corr(data.loc[s.index, "xbb"])),
        )
        .reset_index()
    )
    print(f"\n{prefix}: macro/asset validation by dominant shock:")
    print(macro_asset.round(4))
    macro_asset.to_csv(OUT / f"{prefix}_validation_by_dominant_shock.csv", index=False)

    signed = (
        data.groupby("signed_regime")
        .agg(
            months=("signed_regime", "size"),
            gdp_growth_mean=("gdp_growth", "mean"),
            inflation_mean=("inflation_yoy", "mean"),
            policy_rate_mean=("policy_rate", "mean"),
            xic_monthly_mean=("xic", "mean"),
            xbb_monthly_mean=("xbb", "mean"),
            xic_xbb_corr=("xic", lambda s: s.corr(data.loc[s.index, "xbb"])),
        )
        .sort_values("months", ascending=False)
        .reset_index()
    )
    print(f"\n{prefix}: validation by signed regime:")
    print(signed.round(4))
    signed.to_csv(OUT / f"{prefix}_validation_by_signed_regime.csv", index=False)

    stress = data.loc["2021-01-01":"2022-12-31", [
        "gdp_growth",
        "inflation_yoy",
        "policy_rate",
        "xic",
        "xbb",
        "dominant_shock",
        "dominant_shock_value",
        "signed_regime",
    ]]
    print(f"\n{prefix}: 2021-2022 regime timeline:")
    print(stress.round(4))
    stress.to_csv(OUT / f"{prefix}_2021_2022_regime_timeline.csv")


def markov_inflation_regimes(macro: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate a two-state Markov-switching model on YoY inflation.

    The higher mean-inflation state is labelled high_inflation. The returned
    frame contains smoothed probabilities and the hard regime label for each
    month.
    """
    inflation = macro["inflation_yoy"].dropna()
    model = MarkovRegression(
        inflation,
        k_regimes=2,
        trend="c",
        switching_variance=True,
    )
    res = model.fit(search_reps=20, em_iter=20, disp=False)

    probs = res.smoothed_marginal_probabilities.copy()
    probs.columns = [f"state_{c}_prob" for c in probs.columns]
    means = {}
    for col in probs.columns:
        state = int(str(col).split("_")[1])
        means[state] = inflation[probs[col] >= 0.5].mean()
    high_state = max(means, key=means.get)
    high_col = f"state_{high_state}_prob"

    out = probs.copy()
    out["high_inflation_prob"] = out[high_col]
    out["inflation_regime"] = np.where(
        out["high_inflation_prob"] >= 0.5,
        "high_inflation",
        "low_inflation",
    )
    out["inflation_yoy"] = inflation
    out.attrs["model_summary"] = str(res.summary())
    out.attrs["state_means"] = means
    out.attrs["high_state"] = high_state
    return out


def markov_supply_shock_regimes(structural: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate a two-state Markov-switching model on the SVAR supply-shock series.

    This answers a different question from a Markov model on inflation itself:
    is the economy in a persistent state of unusually large/elevated supply
    shocks, or are supply shocks small and well behaved?
    """
    shock = structural["supply_inflation"].dropna().rename("supply_inflation_shock")
    model = MarkovRegression(
        shock,
        k_regimes=2,
        trend="c",
        switching_variance=True,
    )
    res = model.fit(search_reps=30, em_iter=30, disp=False)

    smoothed = res.smoothed_marginal_probabilities.copy()
    smoothed.columns = [f"state_{c}_smoothed_prob" for c in smoothed.columns]
    filtered = res.filtered_marginal_probabilities.copy()
    filtered.columns = [f"state_{c}_filtered_prob" for c in filtered.columns]

    state_stats = {}
    for col in smoothed.columns:
        state = int(str(col).split("_")[1])
        subset = shock[smoothed[col] >= 0.5]
        state_stats[state] = {
            "mean": subset.mean(),
            "std": subset.std(ddof=1),
            "abs_mean": abs(subset.mean()),
            "obs": len(subset),
        }

    # High-pressure supply state: elevated positive level and/or volatility.
    # The score deliberately rewards both a high average shock and clustering
    # volatility, matching the portfolio question better than inflation alone.
    high_state = max(state_stats, key=lambda s: state_stats[s]["mean"] + state_stats[s]["std"])
    high_smoothed_col = f"state_{high_state}_smoothed_prob"
    high_filtered_col = f"state_{high_state}_filtered_prob"

    out = smoothed.join(filtered, how="inner")
    out["high_pressure_supply_prob"] = out[high_smoothed_col]
    out["supply_shock_regime"] = np.where(
        out["high_pressure_supply_prob"] >= 0.5,
        "high_pressure_supply",
        "normal_supply",
    )
    out["filtered_high_pressure_supply_prob"] = out[high_filtered_col]
    out["filtered_supply_shock_regime"] = np.where(
        out["filtered_high_pressure_supply_prob"] >= 0.5,
        "high_pressure_supply",
        "normal_supply",
    )
    out["supply_inflation_shock"] = shock
    out.attrs["model_summary"] = str(res.summary())
    out.attrs["state_stats"] = state_stats
    out.attrs["high_state"] = high_state
    return out


def markov_portfolio_tables(
    markov: pd.DataFrame,
    returns: pd.DataFrame,
    signed_regimes: pd.DataFrame,
) -> None:
    """Output inflation-regime correlation, 60/40 stats, and SVAR overlap."""
    x = returns.join(markov[["inflation_regime", "high_inflation_prob", "inflation_yoy"]], how="inner")
    x["portfolio_60_40"] = 0.6 * x["xic"] + 0.4 * x["xbb"]

    corr_rows = []
    perf_rows = []
    for regime, g in x.groupby("inflation_regime"):
        corr_rows.append(
            {
                "inflation_regime": regime,
                "obs": len(g),
                "avg_inflation": g["inflation_yoy"].mean(),
                "xic_xbb_corr": g["xic"].corr(g["xbb"]),
            }
        )
        perf_rows.append({"inflation_regime": regime, **perf_stats(g["portfolio_60_40"])})

    corr = pd.DataFrame(corr_rows)
    perf = pd.DataFrame(perf_rows)
    print("\nMarkov inflation regime correlation:")
    print(corr.round(4))
    print("\nMarkov inflation regime 60/40 performance:")
    print(perf.round(4))
    corr.to_csv(OUT / "markov_inflation_regime_correlation.csv", index=False)
    perf.to_csv(OUT / "markov_inflation_regime_60_40_performance.csv", index=False)

    overlap = markov.join(signed_regimes[["signed_regime"]], how="inner")
    table = pd.crosstab(overlap["inflation_regime"], overlap["signed_regime"])
    share = pd.crosstab(overlap["inflation_regime"], overlap["signed_regime"], normalize="index")
    print("\nMarkov/SVAR regime overlap counts:")
    print(table)
    print("\nMarkov/SVAR regime overlap shares:")
    print(share.round(3))
    table.to_csv(OUT / "markov_svar_overlap_counts.csv")
    share.to_csv(OUT / "markov_svar_overlap_shares.csv")


def markov_supply_portfolio_tables(
    markov_supply: pd.DataFrame,
    returns: pd.DataFrame,
    signed_regimes: pd.DataFrame,
    regime_col: str = "supply_shock_regime",
    prob_col: str = "high_pressure_supply_prob",
    prefix: str = "markov_supply",
) -> None:
    """Output portfolio statistics for persistent supply-shock regimes."""
    x = returns.join(
        markov_supply[
            [
                regime_col,
                prob_col,
                "supply_inflation_shock",
            ]
        ],
        how="inner",
    )
    x = x.rename(columns={regime_col: "supply_shock_regime", prob_col: "high_pressure_supply_prob"})
    x["portfolio_60_40"] = 0.6 * x["xic"] + 0.4 * x["xbb"]

    rows = []
    perf_rows = []
    for regime, g in x.groupby("supply_shock_regime"):
        rows.append(
            {
                "supply_shock_regime": regime,
                "obs": len(g),
                "avg_supply_shock": g["supply_inflation_shock"].mean(),
                "supply_shock_vol": g["supply_inflation_shock"].std(ddof=1),
                "avg_high_pressure_prob": g["high_pressure_supply_prob"].mean(),
                "xic_xbb_corr": g["xic"].corr(g["xbb"]),
            }
        )
        perf_rows.append({"supply_shock_regime": regime, **perf_stats(g["portfolio_60_40"])})

    corr = pd.DataFrame(rows)
    perf = pd.DataFrame(perf_rows)
    print("\nMarkov supply-shock regime correlation:")
    print(corr.round(4))
    print("\nMarkov supply-shock regime 60/40 performance:")
    print(perf.round(4))
    corr.to_csv(OUT / f"{prefix}_shock_regime_correlation.csv", index=False)
    perf.to_csv(OUT / f"{prefix}_shock_regime_60_40_performance.csv", index=False)

    overlap = markov_supply[[regime_col]].rename(columns={regime_col: "supply_shock_regime"}).join(
        signed_regimes[["signed_regime"]],
        how="inner",
    )
    table = pd.crosstab(overlap["supply_shock_regime"], overlap["signed_regime"])
    share = pd.crosstab(overlap["supply_shock_regime"], overlap["signed_regime"], normalize="index")
    print("\nMarkov supply-shock/SVAR signed-regime overlap counts:")
    print(table)
    print("\nMarkov supply-shock/SVAR signed-regime overlap shares:")
    print(share.round(3))
    table.to_csv(OUT / f"{prefix}_svar_overlap_counts.csv")
    share.to_csv(OUT / f"{prefix}_svar_overlap_shares.csv")

    stress = x.loc["2021-01-01":"2023-12-31", [
        "supply_inflation_shock",
        "high_pressure_supply_prob",
        "supply_shock_regime",
        "xic",
        "xbb",
        "portfolio_60_40",
    ]]
    print("\nMarkov supply-shock regime timeline, 2021-2023:")
    print(stress.round(4))
    stress.to_csv(OUT / f"{prefix}_2021_2023_timeline.csv")


def fit_garch_11(r: pd.Series) -> tuple[pd.Series, pd.Series, dict[str, float]]:
    """
    Fit a simple Gaussian GARCH(1,1) model and return conditional volatility
    plus standardized residuals.

    Returns are scaled to percent units for numerical stability, then scaled
    back to return units in the output volatility.
    """
    y = (r.dropna() * 100).astype(float)
    eps = y - y.mean()
    var0 = float(eps.var(ddof=1))

    def unpack(theta: np.ndarray) -> tuple[float, float, float]:
        omega = math.exp(theta[0])
        alpha = 1 / (1 + math.exp(-theta[1]))
        beta_raw = 1 / (1 + math.exp(-theta[2]))
        beta = beta_raw * (0.999 - alpha)
        return omega, alpha, beta

    def nll(theta: np.ndarray) -> float:
        omega, alpha, beta = unpack(theta)
        h = np.empty(len(eps))
        h[0] = max(var0, 1e-8)
        vals = eps.to_numpy()
        for t in range(1, len(vals)):
            h[t] = omega + alpha * vals[t - 1] ** 2 + beta * h[t - 1]
            h[t] = max(h[t], 1e-10)
        return 0.5 * np.sum(np.log(h) + vals**2 / h)

    init_alpha = 0.08
    init_beta = 0.88
    init_omega = max(var0 * (1 - init_alpha - init_beta), 1e-6)
    init = np.array(
        [
            math.log(init_omega),
            math.log(init_alpha / (1 - init_alpha)),
            math.log((init_beta / (0.999 - init_alpha)) / (1 - init_beta / (0.999 - init_alpha))),
        ]
    )
    res = minimize(nll, init, method="Nelder-Mead", options={"maxiter": 10000})
    if not res.success:
        res = minimize(nll, init, method="BFGS", options={"maxiter": 10000})
    omega, alpha, beta = unpack(res.x)

    vals = eps.to_numpy()
    h = np.empty(len(vals))
    h[0] = max(var0, 1e-8)
    for t in range(1, len(vals)):
        h[t] = max(omega + alpha * vals[t - 1] ** 2 + beta * h[t - 1], 1e-10)

    vol = pd.Series(np.sqrt(h) / 100, index=y.index, name=f"{r.name}_garch_vol")
    z = pd.Series(vals / np.sqrt(h), index=y.index, name=f"{r.name}_std_resid")
    params = {
        "omega": omega,
        "alpha": alpha,
        "beta": beta,
        "persistence": alpha + beta,
        "success": bool(res.success),
        "nll": float(res.fun),
    }
    return vol, z, params


def fit_dcc_garch(returns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fit a two-asset DCC-GARCH model to XIC/XBB monthly returns.

    Output includes conditional volatilities and the dynamic conditional
    correlation series.
    """
    r = returns[["xic", "xbb"]].dropna()
    xic_vol, xic_z, xic_params = fit_garch_11(r["xic"].rename("xic"))
    xbb_vol, xbb_z, xbb_params = fit_garch_11(r["xbb"].rename("xbb"))
    z = pd.concat([xic_z.rename("xic"), xbb_z.rename("xbb")], axis=1).dropna()
    z_values = z.to_numpy()
    qbar = np.cov(z_values.T)

    def unpack(theta: np.ndarray) -> tuple[float, float]:
        a = 1 / (1 + math.exp(-theta[0]))
        b_raw = 1 / (1 + math.exp(-theta[1]))
        b = b_raw * (0.999 - a)
        return a, b

    def dcc_nll(theta: np.ndarray) -> float:
        a, b = unpack(theta)
        q = qbar.copy()
        total = 0.0
        for t in range(1, len(z_values)):
            prev = z_values[t - 1 : t].T @ z_values[t - 1 : t]
            q = (1 - a - b) * qbar + a * prev + b * q
            d = np.sqrt(np.diag(q))
            rmat = q / np.outer(d, d)
            rmat = np.clip(rmat, -0.999, 0.999)
            np.fill_diagonal(rmat, 1.0)
            sign, logdet = np.linalg.slogdet(rmat)
            if sign <= 0:
                return 1e12
            inv_r = np.linalg.inv(rmat)
            e = z_values[t]
            total += 0.5 * (logdet + e @ inv_r @ e)
        return float(total)

    init = np.array([math.log(0.03 / 0.97), math.log(0.93 / (0.999 - 0.03 - 0.93))])
    res = minimize(dcc_nll, init, method="Nelder-Mead", options={"maxiter": 10000})
    if not res.success:
        res = minimize(dcc_nll, init, method="BFGS", options={"maxiter": 10000})
    a, b = unpack(res.x)

    q = qbar.copy()
    corr = np.empty(len(z_values))
    corr[0] = qbar[0, 1] / math.sqrt(qbar[0, 0] * qbar[1, 1])
    for t in range(1, len(z_values)):
        prev = z_values[t - 1 : t].T @ z_values[t - 1 : t]
        q = (1 - a - b) * qbar + a * prev + b * q
        d = np.sqrt(np.diag(q))
        rmat = q / np.outer(d, d)
        corr[t] = rmat[0, 1]

    dcc = pd.DataFrame(
        {
            "dcc_corr": corr,
            "xic_garch_vol": xic_vol.reindex(z.index),
            "xbb_garch_vol": xbb_vol.reindex(z.index),
        },
        index=z.index,
    )
    params = pd.DataFrame(
        [
            {"component": "xic_garch", **xic_params},
            {"component": "xbb_garch", **xbb_params},
            {
                "component": "dcc",
                "alpha": a,
                "beta": b,
                "persistence": a + b,
                "success": bool(res.success),
                "nll": float(res.fun),
            },
        ]
    )
    return dcc, params


def dcc_summary_tables(
    dcc: pd.DataFrame,
    returns: pd.DataFrame,
    markov_supply: pd.DataFrame,
    signed_regimes: pd.DataFrame,
    regime_col: str = "supply_shock_regime",
    prob_col: str = "high_pressure_supply_prob",
    prefix: str = "dcc_garch",
) -> None:
    """Summarize DCC correlations by macro regime and stress windows."""
    dcc.to_csv(OUT / f"{prefix}_xic_xbb_correlation.csv")

    periods = {
        "Full sample": ("2002-03-01", None),
        "Pre-COVID low-rate era": ("2010-01-01", "2019-12-31"),
        "COVID shock": ("2020-01-01", "2020-12-31"),
        "Inflation/rate shock": ("2021-01-01", "2023-12-31"),
        "Post-2023": ("2024-01-01", None),
    }
    period_rows = []
    for label, (start, end) in periods.items():
        x = dcc.loc[start:end] if end else dcc.loc[start:]
        if len(x):
            period_rows.append(
                {
                    "period": label,
                    "obs": len(x),
                    "avg_dcc_corr": x["dcc_corr"].mean(),
                    "min_dcc_corr": x["dcc_corr"].min(),
                    "max_dcc_corr": x["dcc_corr"].max(),
                }
            )
    period_summary = pd.DataFrame(period_rows)
    print("\nDCC-GARCH correlation by period:")
    print(period_summary.round(4))
    period_summary.to_csv(OUT / f"{prefix}_correlation_by_period.csv", index=False)

    supply = dcc.join(markov_supply[[regime_col, prob_col]], how="inner").rename(
        columns={regime_col: "supply_shock_regime", prob_col: "high_pressure_supply_prob"}
    )
    supply_summary = (
        supply.groupby("supply_shock_regime")
        .agg(
            obs=("dcc_corr", "size"),
            avg_dcc_corr=("dcc_corr", "mean"),
            median_dcc_corr=("dcc_corr", "median"),
            min_dcc_corr=("dcc_corr", "min"),
            max_dcc_corr=("dcc_corr", "max"),
            avg_high_pressure_prob=("high_pressure_supply_prob", "mean"),
        )
        .reset_index()
    )
    print("\nDCC-GARCH correlation by Markov supply-shock regime:")
    print(supply_summary.round(4))
    supply_summary.to_csv(OUT / f"{prefix}_by_markov_supply_regime.csv", index=False)

    signed = dcc.join(signed_regimes[["signed_regime"]], how="inner")
    signed_summary = (
        signed.groupby("signed_regime")
        .agg(
            obs=("dcc_corr", "size"),
            avg_dcc_corr=("dcc_corr", "mean"),
            median_dcc_corr=("dcc_corr", "median"),
            min_dcc_corr=("dcc_corr", "min"),
            max_dcc_corr=("dcc_corr", "max"),
        )
        .sort_values("avg_dcc_corr", ascending=False)
        .reset_index()
    )
    print("\nDCC-GARCH correlation by signed SVAR regime:")
    print(signed_summary.round(4))
    signed_summary.to_csv(OUT / f"{prefix}_by_signed_svar_regime.csv", index=False)

    timeline = dcc.join(returns, how="inner").join(
        markov_supply[[regime_col, prob_col]],
        how="inner",
    ).rename(columns={regime_col: "supply_shock_regime", prob_col: "high_pressure_supply_prob"}).loc[
        "2021-01-01":"2023-12-31"
    ]
    print("\nDCC-GARCH timeline, 2021-2023:")
    print(timeline.round(4))
    timeline.to_csv(OUT / f"{prefix}_2021_2023_timeline.csv")


def fetch_portfolio_assets() -> pd.DataFrame:
    """
    Fetch investable public ETF proxies for portfolio construction.

    Core Canadian assets:
    - XIC.TO: Canadian equities
    - XBB.TO: broad Canadian bonds
    - XSB.TO: short-term Canadian bonds
    - CGL-C.TO: Canadian-listed gold bullion, unhedged to CAD
    - CGL.TO: Canadian-listed gold bullion, CAD hedged
    - DBC: broad commodities ETF, USD-listed
    - FXC: CAD/USD proxy, USD-listed

    Some symbols may have shorter histories. The portfolio tables use the
    common available sample after joining with regimes.
    """
    symbols = {
        "xic": "XIC.TO",
        "xbb": "XBB.TO",
        "xsb": "XSB.TO",
        "gold_unhedged": "CGL-C.TO",
        "gold_hedged": "CGL.TO",
        "commodities": "DBC",
        "cad": "FXC",
    }
    prices = {}
    errors = {}
    for name, symbol in symbols.items():
        try:
            prices[name] = yahoo_monthly_adjclose(symbol)
        except Exception as exc:
            errors[name] = f"{symbol}: {exc}"

    if errors:
        print("\nPortfolio asset fetch issues:")
        for msg in errors.values():
            print("  -", msg)

    asset_returns = pd.concat({k: monthly_returns(v) for k, v in prices.items()}, axis=1).dropna(how="all")
    asset_returns.to_csv(OUT / "portfolio_asset_monthly_returns.csv")
    return asset_returns


def make_portfolios(asset_returns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct practical portfolio variants from available assets."""
    candidates = {
        "60_40_xic_xbb": {"xic": 0.60, "xbb": 0.40},
        "70_30_xic_xbb": {"xic": 0.70, "xbb": 0.30},
        "70_20_10_gold_unhedged": {"xic": 0.70, "xbb": 0.20, "gold_unhedged": 0.10},
        "70_20_10_gold_hedged": {"xic": 0.70, "xbb": 0.20, "gold_hedged": 0.10},
        "70_15_10_5_gold_unhedged_commod": {"xic": 0.70, "xbb": 0.15, "gold_unhedged": 0.10, "commodities": 0.05},
        "70_15_10_5_gold_hedged_commod": {"xic": 0.70, "xbb": 0.15, "gold_hedged": 0.10, "commodities": 0.05},
        "65_20_10_5_short_gold_unhedged": {"xic": 0.65, "xbb": 0.20, "xsb": 0.10, "gold_unhedged": 0.05},
        "65_20_10_5_short_gold_hedged": {"xic": 0.65, "xbb": 0.20, "xsb": 0.10, "gold_hedged": 0.05},
        "75_15_10_gold_unhedged": {"xic": 0.75, "xbb": 0.15, "gold_unhedged": 0.10},
        "75_15_10_gold_hedged": {"xic": 0.75, "xbb": 0.15, "gold_hedged": 0.10},
        "60_30_10_gold_unhedged": {"xic": 0.60, "xbb": 0.30, "gold_unhedged": 0.10},
        "60_30_10_gold_hedged": {"xic": 0.60, "xbb": 0.30, "gold_hedged": 0.10},
        "50_30_20_gold_unhedged": {"xic": 0.50, "xbb": 0.30, "gold_unhedged": 0.20},
        "50_30_20_gold_hedged": {"xic": 0.50, "xbb": 0.30, "gold_hedged": 0.20},
        "50_25_15_10_short_gold_unhedged": {"xic": 0.50, "xbb": 0.25, "xsb": 0.15, "gold_unhedged": 0.10},
        "50_25_15_10_short_gold_hedged": {"xic": 0.50, "xbb": 0.25, "xsb": 0.15, "gold_hedged": 0.10},
        "55_25_10_10_short_gold_unhedged": {"xic": 0.55, "xbb": 0.25, "xsb": 0.10, "gold_unhedged": 0.10},
        "55_25_10_10_short_gold_hedged": {"xic": 0.55, "xbb": 0.25, "xsb": 0.10, "gold_hedged": 0.10},
        "55_20_15_10_short_gold_unhedged": {"xic": 0.55, "xbb": 0.20, "xsb": 0.15, "gold_unhedged": 0.10},
        "55_20_15_10_short_gold_hedged": {"xic": 0.55, "xbb": 0.20, "xsb": 0.15, "gold_hedged": 0.10},
        "60_25_10_5_commod_cad": {"xic": 0.60, "xbb": 0.25, "commodities": 0.10, "cad": 0.05},
        "60_20_20_short_bonds": {"xic": 0.60, "xbb": 0.20, "xsb": 0.20},
        "50_25_10_10_5_real_asset_mix_unhedged": {
            "xic": 0.50,
            "xbb": 0.25,
            "xsb": 0.10,
            "gold_unhedged": 0.10,
            "commodities": 0.05,
        },
        "50_25_10_10_5_real_asset_mix_hedged": {
            "xic": 0.50,
            "xbb": 0.25,
            "xsb": 0.10,
            "gold_hedged": 0.10,
            "commodities": 0.05,
        },
        "50_20_15_10_5_real_asset_mix_unhedged": {
            "xic": 0.50,
            "xbb": 0.20,
            "xsb": 0.15,
            "gold_unhedged": 0.10,
            "commodities": 0.05,
        },
        "50_20_15_10_5_real_asset_mix_hedged": {
            "xic": 0.50,
            "xbb": 0.20,
            "xsb": 0.15,
            "gold_hedged": 0.10,
            "commodities": 0.05,
        },
        "45_25_15_10_5_real_asset_mix_unhedged": {
            "xic": 0.45,
            "xbb": 0.25,
            "xsb": 0.15,
            "gold_unhedged": 0.10,
            "commodities": 0.05,
        },
        "45_25_15_10_5_real_asset_mix_hedged": {
            "xic": 0.45,
            "xbb": 0.25,
            "xsb": 0.15,
            "gold_hedged": 0.10,
            "commodities": 0.05,
        },
    }
    rows = []
    portfolio_returns = {}
    for name, weights in candidates.items():
        available = [asset for asset in weights if asset in asset_returns.columns]
        missing = [asset for asset in weights if asset not in asset_returns.columns]
        if missing:
            print(f"Skipping {name}; missing assets: {', '.join(missing)}")
            continue
        w = pd.Series(weights, dtype=float)
        r = asset_returns[available].dropna()
        pr = r.mul(w[available], axis=1).sum(axis=1)
        pr.name = name
        portfolio_returns[name] = pr
        for asset, weight in weights.items():
            rows.append({"portfolio": name, "asset": asset, "weight": weight})

    weights_table = pd.DataFrame(rows)
    portfolios = pd.concat(portfolio_returns.values(), axis=1).dropna(how="all")
    weights_table.to_csv(OUT / "portfolio_weights.csv", index=False)
    portfolios.to_csv(OUT / "portfolio_monthly_returns.csv")
    return portfolios, weights_table


def portfolio_regime_tables(
    portfolios: pd.DataFrame,
    asset_returns: pd.DataFrame,
    markov_supply: pd.DataFrame,
    regime_col: str = "supply_shock_regime",
    prob_col: str = "high_pressure_supply_prob",
    prefix: str = "portfolio",
) -> None:
    """Summarize practical portfolio variants by supply-shock regime."""
    data = portfolios.join(
        markov_supply[[regime_col, prob_col]],
        how="inner",
    ).rename(columns={regime_col: "supply_shock_regime", prob_col: "high_pressure_supply_prob"})
    rf = asset_returns["xsb"].reindex(data.index) if "xsb" in asset_returns.columns else pd.Series(0, index=data.index)

    rows = []
    for portfolio in portfolios.columns:
        for regime, g in data[[portfolio, "supply_shock_regime"]].dropna().groupby("supply_shock_regime"):
            r = g[portfolio]
            excess = r - rf.reindex(r.index).fillna(0)
            stats = perf_stats(r)
            rows.append(
                {
                    "portfolio": portfolio,
                    "supply_shock_regime": regime,
                    **stats,
                    "sharpe_vs_xsb": (excess.mean() * 12) / (excess.std(ddof=1) * math.sqrt(12))
                    if excess.std(ddof=1) > 0
                    else np.nan,
                }
            )
    summary = pd.DataFrame(rows)

    base = summary[summary["portfolio"] == "60_40_xic_xbb"][
        ["supply_shock_regime", "ann_return", "ann_vol", "max_drawdown", "hist_var_5"]
    ].rename(
        columns={
            "ann_return": "base_ann_return",
            "ann_vol": "base_ann_vol",
            "max_drawdown": "base_max_drawdown",
            "hist_var_5": "base_hist_var_5",
        }
    )
    summary = summary.merge(base, on="supply_shock_regime", how="left")
    summary["ann_return_diff_vs_60_40"] = summary["ann_return"] - summary["base_ann_return"]
    summary["ann_vol_diff_vs_60_40"] = summary["ann_vol"] - summary["base_ann_vol"]
    summary["max_drawdown_diff_vs_60_40"] = summary["max_drawdown"] - summary["base_max_drawdown"]
    summary["hist_var_5_diff_vs_60_40"] = summary["hist_var_5"] - summary["base_hist_var_5"]

    print("\nPortfolio performance by Markov supply-shock regime:")
    display_cols = [
        "portfolio",
        "supply_shock_regime",
        "obs",
        "ann_return",
        "ann_vol",
        "max_drawdown",
        "worst_month",
        "hist_var_5",
        "sharpe_vs_xsb",
        "ann_return_diff_vs_60_40",
        "ann_vol_diff_vs_60_40",
        "max_drawdown_diff_vs_60_40",
    ]
    print(summary[display_cols].round(4))
    summary.to_csv(OUT / f"{prefix}_performance_by_supply_regime.csv", index=False)

    high_pressure = summary[summary["supply_shock_regime"] == "high_pressure_supply"].copy()
    high_pressure = high_pressure.sort_values(["max_drawdown", "ann_vol"], ascending=[False, True])
    print("\nHigh-pressure supply regime portfolio ranking:")
    print(high_pressure[display_cols].round(4))
    high_pressure.to_csv(OUT / f"{prefix}_ranking_high_pressure_supply.csv", index=False)

    # Asset-level diagnostics explain which substitutes helped or hurt.
    asset_data = asset_returns.join(
        markov_supply[[regime_col, prob_col]],
        how="inner",
    ).rename(columns={regime_col: "supply_shock_regime", prob_col: "high_pressure_supply_prob"})
    asset_rows = []
    for asset in [c for c in asset_returns.columns if c in asset_data.columns]:
        for regime, g in asset_data[[asset, "supply_shock_regime"]].dropna().groupby("supply_shock_regime"):
            asset_rows.append({"asset": asset, "supply_shock_regime": regime, **perf_stats(g[asset])})
    asset_summary = pd.DataFrame(asset_rows)
    print("\nAsset-level performance by Markov supply-shock regime:")
    print(asset_summary.round(4))
    asset_summary.to_csv(OUT / f"{prefix}_asset_performance_by_supply_regime.csv", index=False)

    # Worst high-pressure supply months for the baseline portfolio.
    baseline = data[["60_40_xic_xbb", "supply_shock_regime", "high_pressure_supply_prob"]].dropna()
    worst = baseline[baseline["supply_shock_regime"] == "high_pressure_supply"].nsmallest(
        12, "60_40_xic_xbb"
    )
    print("\nWorst 60/40 months in high-pressure supply regime:")
    print(worst.round(4))
    worst.to_csv(OUT / f"{prefix}_worst_60_40_months_high_pressure_supply.csv")


def robustness_macro_changes(macro: pd.DataFrame) -> pd.DataFrame:
    """
    Robustness version using less persistent inflation and policy variables.

    Keeps GDP growth as the activity variable, but uses changes in YoY inflation
    and changes in the policy rate. The columns are renamed to the baseline names
    so the same sign-restricted SVAR helper can be reused.
    """
    x = pd.DataFrame(
        {
            "gdp_growth": macro["gdp_growth"],
            "inflation_yoy": macro["inflation_yoy"].diff(),
            "policy_rate": macro["policy_rate"].diff(),
        },
        index=macro.index,
    ).dropna()
    return x


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main() -> None:
    print("Pulling market data...")
    xic = yahoo_monthly_adjclose("XIC.TO")
    xbb = yahoo_monthly_adjclose("XBB.TO")
    returns = pd.concat([monthly_returns(xic), monthly_returns(xbb)], axis=1).dropna()
    returns.columns = ["xic", "xbb"]
    returns.to_csv(OUT / "xic_xbb_monthly_returns.csv")

    print("\nMarket sample statistics:")
    stats = pd.DataFrame({"xic": perf_stats(returns["xic"]), "xbb": perf_stats(returns["xbb"])}).T
    print((stats * 100).round(2).assign(obs=stats["obs"].astype(int)))
    stats.to_csv(OUT / "market_sample_statistics.csv")

    print("\nCorrelation by period:")
    period_corr = corr_by_period(returns)
    print(period_corr.round(3))
    period_corr.to_csv(OUT / "correlation_by_period.csv", index=False)

    print("\nPulling macro data...")
    cpi_raw = statcan_table("18100004")
    gdp_raw = statcan_table("36100434")

    cpi = pick_series(
        cpi_raw,
        value_name="cpi",
        filters={
            "GEO": r"^Canada$",
            "Products": r"All-items",
        },
    )
    gdp = pick_series(
        gdp_raw,
        value_name="gdp",
        filters={
            "GEO": r"^Canada$",
            "Seasonal adjustment": r"Seasonally adjusted",
            "Prices": r"chained|constant|2012",
            "North American|Industry": r"All industries",
        },
    )

    # Bank of Canada Valet series V39079 = Target for the overnight rate.
    # This is more stable than selecting the row from the broad StatCan
    # financial-market table.
    policy_daily = boc_valet_series("V39079").rename("policy_rate")
    policy = to_month_end_index(policy_daily.resample("M").last())

    macro = pd.concat(
        [
            to_month_end_index(np.log(gdp).diff().mul(100).rename("gdp_growth")),
            to_month_end_index(cpi.pct_change(12).mul(100).rename("inflation_yoy")),
            policy.rename("policy_rate"),
        ],
        axis=1,
    ).dropna()
    macro = macro.loc[returns.index.min() : returns.index.max()]
    print("\nMacro SVAR input checkpoint:")
    print(
        pd.DataFrame(
            {
                "start": macro.min().index.map(lambda _: macro.index.min()),
                "end": macro.min().index.map(lambda _: macro.index.max()),
                "non_null_obs": macro.notna().sum(),
                "mean": macro.mean(),
                "std": macro.std(),
            }
        )
    )
    if len(macro) <= 12:
        raise ValueError(
            "Macro SVAR input has too few observations after alignment. "
            "Check outputs/macro_monthly_transformed.csv and source selectors."
        )
    macro.to_csv(OUT / "macro_monthly_transformed.csv")

    print("\nEstimating sign-restricted SVAR...")
    structural, regimes = sign_restricted_structural_shocks(macro)
    structural.to_csv(OUT / "svar_structural_shocks.csv")
    regimes.to_csv(OUT / "svar_dominant_regime.csv")
    signed_regime_labels(structural).to_csv(OUT / "svar_signed_regime.csv")

    shock_counts = regimes["dominant_shock"].value_counts().rename_axis("regime").reset_index(name="months")
    print("\nDominant shock counts:")
    print(shock_counts)
    shock_counts.to_csv(OUT / "shock_counts.csv", index=False)

    corr_regime, perf_regime = regime_portfolio_tables(returns, regimes)
    print("\nBond-equity correlation by SVAR regime:")
    print(corr_regime.round(3))
    corr_regime.to_csv(OUT / "correlation_by_svar_regime.csv", index=False)

    print("\n60/40 performance by SVAR regime:")
    print(perf_regime.round(4))
    perf_regime.to_csv(OUT / "portfolio_60_40_by_svar_regime.csv", index=False)

    validation_tables(macro, returns, structural, regimes, "baseline")

    print("\nEstimating robustness SVAR with changes in inflation and policy rate...")
    macro_changes = robustness_macro_changes(macro)
    macro_changes.to_csv(OUT / "macro_monthly_transformed_changes_robustness.csv")
    structural_chg, regimes_chg = sign_restricted_structural_shocks(
        macro_changes,
        lags=6,
        draws=10000,
        seed=11,
    )
    structural_chg.to_csv(OUT / "robustness_svar_structural_shocks.csv")
    regimes_chg.to_csv(OUT / "robustness_svar_dominant_regime.csv")
    signed_regime_labels(structural_chg).to_csv(OUT / "robustness_svar_signed_regime.csv")

    shock_counts_chg = regimes_chg["dominant_shock"].value_counts().rename_axis("regime").reset_index(name="months")
    print("\nRobustness dominant shock counts:")
    print(shock_counts_chg)
    shock_counts_chg.to_csv(OUT / "robustness_shock_counts.csv", index=False)

    corr_regime_chg, perf_regime_chg = regime_portfolio_tables(returns, regimes_chg)
    print("\nRobustness bond-equity correlation by SVAR regime:")
    print(corr_regime_chg.round(3))
    corr_regime_chg.to_csv(OUT / "robustness_correlation_by_svar_regime.csv", index=False)

    print("\nRobustness 60/40 performance by SVAR regime:")
    print(perf_regime_chg.round(4))
    perf_regime_chg.to_csv(OUT / "robustness_portfolio_60_40_by_svar_regime.csv", index=False)

    validation_tables(macro, returns, structural_chg, regimes_chg, "robustness")

    print("\nEstimating Markov-switching regimes on the SVAR supply-shock series...")
    markov_supply = markov_supply_shock_regimes(structural_chg)
    markov_supply.to_csv(OUT / "markov_supply_shock_regimes.csv")
    state_stats_text = []
    for state, vals in markov_supply.attrs["state_stats"].items():
        state_stats_text.append(
            f"state_{state}: "
            f"mean={vals['mean']:.4f}, "
            f"std={vals['std']:.4f}, "
            f"abs_mean={vals['abs_mean']:.4f}, "
            f"obs={vals['obs']}"
        )
    (OUT / "markov_supply_shock_model_summary.txt").write_text(
        markov_supply.attrs["model_summary"]
        + "\n\nState supply-shock stats:\n"
        + "\n".join(state_stats_text)
        + f"\nHigh-pressure supply state: state_{markov_supply.attrs['high_state']}\n",
        encoding="utf-8",
    )
    markov_supply_portfolio_tables(
        markov_supply,
        returns,
        signed_regime_labels(structural_chg),
        regime_col="supply_shock_regime",
        prob_col="high_pressure_supply_prob",
        prefix="markov_supply_smoothed",
    )
    markov_supply_portfolio_tables(
        markov_supply,
        returns,
        signed_regime_labels(structural_chg),
        regime_col="filtered_supply_shock_regime",
        prob_col="filtered_high_pressure_supply_prob",
        prefix="markov_supply_filtered",
    )

    print("\nEstimating DCC-GARCH for XIC/XBB monthly returns...")
    dcc, dcc_params = fit_dcc_garch(returns)
    print("\nDCC-GARCH parameter estimates:")
    print(dcc_params.round(4))
    dcc_params.to_csv(OUT / "dcc_garch_parameters.csv", index=False)
    dcc_summary_tables(
        dcc,
        returns,
        markov_supply,
        signed_regime_labels(structural_chg),
        regime_col="supply_shock_regime",
        prob_col="high_pressure_supply_prob",
        prefix="dcc_garch_smoothed",
    )
    dcc_summary_tables(
        dcc,
        returns,
        markov_supply,
        signed_regime_labels(structural_chg),
        regime_col="filtered_supply_shock_regime",
        prob_col="filtered_high_pressure_supply_prob",
        prefix="dcc_garch_filtered",
    )

    print("\nRunning portfolio construction tests...")
    portfolio_asset_returns = fetch_portfolio_assets()
    portfolios, weights_table = make_portfolios(portfolio_asset_returns)
    print("\nPortfolio weights:")
    print(weights_table)
    portfolio_regime_tables(
        portfolios,
        portfolio_asset_returns,
        markov_supply,
        regime_col="supply_shock_regime",
        prob_col="high_pressure_supply_prob",
        prefix="portfolio_smoothed",
    )
    portfolio_regime_tables(
        portfolios,
        portfolio_asset_returns,
        markov_supply,
        regime_col="filtered_supply_shock_regime",
        prob_col="filtered_high_pressure_supply_prob",
        prefix="portfolio_filtered",
    )

    print("\nEstimating Markov-switching inflation regimes as a secondary comparison...")
    markov = markov_inflation_regimes(macro)
    markov.to_csv(OUT / "markov_inflation_regimes.csv")
    (OUT / "markov_inflation_model_summary.txt").write_text(
        markov.attrs["model_summary"]
        + "\n\nState inflation means:\n"
        + "\n".join(f"state_{k}: {v:.4f}" for k, v in markov.attrs["state_means"].items())
        + f"\nHigh-inflation state: state_{markov.attrs['high_state']}\n",
        encoding="utf-8",
    )
    markov_portfolio_tables(markov, returns, signed_regime_labels(structural_chg))

    print(f"\nDone. Outputs written to: {OUT.resolve()}")


if __name__ == "__main__":
    main()
