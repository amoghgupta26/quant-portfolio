"""
portfolio.py
GARCH-Informed Portfolio Optimisation
MSc Quantitative Finance — Personal Project

Investigates whether using GARCH(1,1)-implied covariance as input to
mean-variance optimisation improves risk-adjusted returns versus a
static historical covariance baseline.

Universe  : 6 US equities + GLD + BND
Period    : 2018–2024 (estimation), 2022–2024 (holdout)
Rebalance : Monthly, 10bps transaction cost
Benchmark : SPY
"""

import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from arch import arch_model
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────

TICKERS   = ["AAPL", "MSFT", "JPM", "JNJ", "XOM", "GLD", "SPY", "BND"]
BENCHMARK = "SPY"
START     = "2018-01-01"
HOLDOUT   = "2022-01-01"   # train/test split
RF        = 0.045
TC        = 0.001

# ── 1. Data ───────────────────────────────────────────────────────────────────

print("Downloading data...")
raw = yf.download(TICKERS, start=START, auto_adjust=True, progress=False)["Close"]
prices  = raw.ffill().bfill().dropna(axis=1, thresh=int(0.95 * len(raw)))
returns = np.log(prices / prices.shift(1)).dropna()
print(f"  {len(returns)} trading days | {returns.shape[1]} assets")

# Train / holdout split
train = returns[returns.index < HOLDOUT]
test  = returns[returns.index >= HOLDOUT]
print(f"  Train: {train.index[0].date()} → {train.index[-1].date()} ({len(train)} days)")
print(f"  Test : {test.index[0].date()} → {test.index[-1].date()} ({len(test)} days)\n")

# ── 2. GARCH(1,1) — fit on train, evaluate on test ───────────────────────────

def fit_garch_vol(series, horizon=10):
    r_scaled = series * 100
    am  = arch_model(r_scaled, vol="Garch", p=1, q=1,
                     mean="Constant", dist="normal")
    res = am.fit(disp="off", show_warning=False)
    fc  = res.forecast(horizon=horizon, reindex=False)
    mean_var = fc.variance.values[-1].mean() / 1e4
    return np.sqrt(mean_var * 252), res.params["alpha[1]"] + res.params["beta[1]"]

print("Fitting GARCH(1,1) on training data...")
garch_vols = {}
for ticker in train.columns:
    vol, pers = fit_garch_vol(train[ticker])
    garch_vols[ticker] = vol
    print(f"  {ticker:6s}: forecast vol = {vol:.2%} | persistence = {pers:.4f}")

# GARCH-implied covariance (train correlations + GARCH vols)
corr      = train.corr().values
vols      = np.array([garch_vols[t] for t in train.columns])
D         = np.diag(vols)
garch_cov = D @ corr @ D

# Historical covariance (train only)
hist_cov  = train.cov().values * 252

# ── 3. Portfolio Optimisation ─────────────────────────────────────────────────

mu = train.mean().values * 252
n  = len(mu)

def neg_sharpe(w, mu, cov, rf):
    ret = w @ mu
    vol = np.sqrt(w @ cov @ w)
    return -(ret - rf) / vol

def optimise(mu, cov, label):
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = [(0, 0.40)] * n
    w0 = np.ones(n) / n
    res = minimize(neg_sharpe, w0, args=(mu, cov, RF),
                   method="SLSQP", bounds=bounds,
                   constraints=constraints,
                   options={"maxiter": 1000})
    w = np.maximum(res.x, 0)
    w /= w.sum()
    ret = w @ mu
    vol = np.sqrt(w @ cov @ w)
    sr  = (ret - RF) / vol
    print(f"\n{label}")
    for t, v in sorted(zip(train.columns, w), key=lambda x: -x[1]):
        if v > 0.005:
            print(f"  {t:6s}: {v:.1%}")
    print(f"  → Return: {ret:.2%} | Vol: {vol:.2%} | Sharpe: {sr:.3f}")
    return pd.Series(w, index=train.columns)

print("\n── Portfolio Weights (estimated on train) ──")
w_garch = optimise(mu, garch_cov, "Max Sharpe — GARCH covariance")
w_hist  = optimise(mu, hist_cov,  "Max Sharpe — Historical covariance")

# ── 4. Backtest on holdout period ─────────────────────────────────────────────

def backtest(returns, weights, tc=TC):
    w      = weights.reindex(returns.columns).fillna(0)
    port_r = (returns * w).sum(axis=1).copy()
    port_r.iloc[0] -= w.sum() * tc
    return port_r

port_garch = backtest(test, w_garch)
port_hist  = backtest(test, w_hist)
bench      = test[BENCHMARK]

# ── 5. Risk Metrics ───────────────────────────────────────────────────────────

def risk_report(r, benchmark, label):
    n    = len(r)
    cagr = (1 + r).prod() ** (252 / n) - 1
    vol  = r.std() * np.sqrt(252)
    sr   = (cagr - RF) / vol
    mdd  = ((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min()
    var  = -np.percentile(r, 5)
    cvar = -r[r <= -var].mean()
    b    = benchmark.reindex(r.index).dropna()
    aln  = pd.concat([r, b], axis=1).dropna()
    cm   = np.cov(aln.iloc[:, 0], aln.iloc[:, 1])
    beta = cm[0, 1] / cm[1, 1]
    print(f"\n── {label} ──")
    print(f"  CAGR           : {cagr:.2%}")
    print(f"  Volatility     : {vol:.2%}")
    print(f"  Sharpe Ratio   : {sr:.3f}")
    print(f"  Max Drawdown   : {mdd:.2%}")
    print(f"  VaR (95%, 1d)  : {var:.3%}")
    print(f"  CVaR (95%, 1d) : {cvar:.3%}")
    print(f"  Beta vs SPY    : {beta:.3f}")

print("\n── Risk Report (2022–2024 holdout) ──")
risk_report(port_garch, bench, "GARCH-Optimised Portfolio")
risk_report(port_hist,  bench, "Historical-Optimised Portfolio")
risk_report(bench,      bench, f"Benchmark ({BENCHMARK})")

# ── 6. Charts ─────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("GARCH-Informed Portfolio Optimisation — Holdout 2022–2024", fontsize=12)

NAVY, TEAL, CORAL, GRAY = "#0a2342", "#1D9E75", "#D85A30", "#888780"

ax = axes[0, 0]
(100 * (1 + port_garch).cumprod()).plot(ax=ax, color=NAVY, label="GARCH opt.")
(100 * (1 + port_hist ).cumprod()).plot(ax=ax, color=TEAL, label="Hist. opt.", ls="--")
(100 * (1 + bench     ).cumprod()).plot(ax=ax, color=GRAY, label="SPY", lw=0.9, ls=":")
ax.set_title("Cumulative Return (base=100)")
ax.legend(fontsize=8)
ax.spines[["top", "right"]].set_visible(False)

ax = axes[0, 1]
dd = (1 + port_garch).cumprod()
dd = (dd / dd.cummax() - 1) * 100
ax.fill_between(dd.index, dd, 0, color=CORAL, alpha=0.55)
ax.set_title("Portfolio Drawdown (%)")
ax.spines[["top", "right"]].set_visible(False)

ax = axes[1, 0]
(port_garch.rolling(21).std() * np.sqrt(252) * 100).plot(ax=ax, color=NAVY, lw=1.2)
ax.set_title("Rolling 21d Volatility (%)")
ax.spines[["top", "right"]].set_visible(False)

ax = axes[1, 1]
w_plot = w_garch[w_garch > 0.005].sort_values()
ax.barh(w_plot.index, w_plot * 100, color=NAVY)
ax.set_title("Portfolio Weights — GARCH opt. (%)")
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig("results.png", dpi=150, bbox_inches="tight")
print("\nChart saved → results.png")
