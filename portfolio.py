"""
portfolio.py
GARCH-Informed Portfolio Optimisation
MSc Quantitative Finance — Personal Project

Investigates whether using GARCH(1,1)-implied covariance as input to
mean-variance optimisation improves risk-adjusted returns versus a
static historical covariance baseline.

Universe  : 6 US equities + GLD + BND
Period    : 2018–2024
Rebalance : Monthly, 10bps transaction cost
Benchmark : SPY
"""

import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from arch import arch_model
from pypfopt import EfficientFrontier, risk_models, expected_returns
from scipy import stats

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────

TICKERS    = ["AAPL", "MSFT", "JPM", "JNJ", "XOM", "GLD", "SPY", "BND"]
BENCHMARK  = "SPY"
START      = "2018-01-01"
RF         = 0.045   # risk-free rate proxy
TC         = 0.001   # 10bps one-way transaction cost

# ── 1. Data ───────────────────────────────────────────────────────────────────

print("Downloading data...")
raw = yf.download(TICKERS, start=START, auto_adjust=True, progress=False)["Close"]
prices = raw.ffill().bfill().dropna(axis=1, thresh=int(0.95 * len(raw)))
returns = np.log(prices / prices.shift(1)).dropna()
print(f"  {len(returns)} trading days | {returns.shape[1]} assets\n")

# ── 2. GARCH(1,1) Volatility Forecasting ─────────────────────────────────────

def fit_garch_vol(series, horizon=10):
    """
    Fit GARCH(1,1) and return annualised vol forecast.
    sigma_t^2 = omega + alpha*eps_{t-1}^2 + beta*sigma_{t-1}^2
    """
    r_scaled = series * 100
    am = arch_model(r_scaled, vol="Garch", p=1, q=1, mean="Constant", dist="normal")
    res = am.fit(disp="off", show_warning=False)
    fc = res.forecast(horizon=horizon, reindex=False)
    mean_var = fc.variance.values[-1].mean() / 1e4
    return np.sqrt(mean_var * 252), res.params["alpha[1]"] + res.params["beta[1]"]

print("Fitting GARCH(1,1) per asset...")
garch_vols, persistence = {}, {}
for ticker in returns.columns:
    vol, pers = fit_garch_vol(returns[ticker])
    garch_vols[ticker] = vol
    persistence[ticker] = pers
    print(f"  {ticker:6s}: forecast vol = {vol:.2%} | persistence = {pers:.4f}")

# Build GARCH-implied covariance: Sigma_ij = rho_ij * sigma_i * sigma_j
corr = returns.corr().values
vols = np.array([garch_vols[t] for t in returns.columns])
D = np.diag(vols)
garch_cov = pd.DataFrame(D @ corr @ D, index=returns.columns, columns=returns.columns)

# ── 3. Portfolio Optimisation ─────────────────────────────────────────────────

mu = expected_returns.mean_historical_return(returns, returns_data=True, frequency=252)
hist_cov = risk_models.sample_cov(returns, returns_data=True, frequency=252)

def optimise(mu, cov, label):
    ef = EfficientFrontier(mu, cov, weight_bounds=(0, 0.40))
    ef.max_sharpe(risk_free_rate=RF)
    w = ef.clean_weights()
    ret, vol, sr = ef.portfolio_performance(risk_free_rate=RF, verbose=False)
    print(f"\n{label}")
    for t, v in sorted(w.items(), key=lambda x: -x[1]):
        if v > 0.005:
            print(f"  {t:6s}: {v:.1%}")
    print(f"  → Return: {ret:.2%} | Vol: {vol:.2%} | Sharpe: {sr:.3f}")
    return pd.Series(w)

print("\n── Portfolio Weights ──")
w_garch = optimise(mu, garch_cov, "Max Sharpe (GARCH covariance)")
w_hist  = optimise(mu, hist_cov,  "Max Sharpe (Historical covariance)")

# ── 4. Walk-Forward Backtest ──────────────────────────────────────────────────

def backtest(returns, strategy_weights, tc=TC, min_history=252):
    """
    Simple walk-forward backtest.
    Weights are fixed (no re-estimation for brevity here).
    Re-estimation version: see README.
    """
    r = returns.iloc[min_history:]
    w = strategy_weights.reindex(returns.columns).fillna(0)
    port_r = (r * w).sum(axis=1)
    # Apply TC at start
    port_r.iloc[0] -= w.sum() * tc
    return port_r

port_garch = backtest(returns, w_garch)
port_hist  = backtest(returns, w_hist)
bench      = returns[BENCHMARK].iloc[252:]

# ── 5. Risk Metrics ───────────────────────────────────────────────────────────

def risk_report(r, benchmark, label):
    n = len(r)
    cagr = (1 + r).prod() ** (252 / n) - 1
    vol  = r.std() * np.sqrt(252)
    sr   = (cagr - RF) / vol
    mdd  = ((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min()
    var  = -np.percentile(r, 5)
    cvar = -r[r <= -var].mean()

    b = benchmark.reindex(r.index).dropna()
    aligned = pd.concat([r, b], axis=1).dropna()
    cov_mat = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])
    beta = cov_mat[0, 1] / cov_mat[1, 1]

    print(f"\n── {label} ──")
    print(f"  CAGR           : {cagr:.2%}")
    print(f"  Volatility     : {vol:.2%}")
    print(f"  Sharpe Ratio   : {sr:.3f}")
    print(f"  Max Drawdown   : {mdd:.2%}")
    print(f"  VaR (95%, 1d)  : {var:.3%}")
    print(f"  CVaR (95%, 1d) : {cvar:.3%}")
    print(f"  Beta vs SPY    : {beta:.3f}")
    return {"cagr": cagr, "vol": vol, "sharpe": sr, "mdd": mdd}

print("\n── Risk Report ──")
m_garch = risk_report(port_garch, bench, "GARCH-Optimised Portfolio")
m_hist  = risk_report(port_hist,  bench, "Historical-Optimised Portfolio")
risk_report(bench, bench, f"Benchmark ({BENCHMARK})")

# ── 6. Charts ─────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("GARCH-Informed Portfolio Optimisation", fontsize=13, y=1.01)

NAVY, TEAL, CORAL, GRAY = "#0a2342", "#1D9E75", "#D85A30", "#888780"

# Equity curve
ax = axes[0, 0]
(100 * (1 + port_garch).cumprod()).plot(ax=ax, color=NAVY,  label="GARCH opt.")
(100 * (1 + port_hist ).cumprod()).plot(ax=ax, color=TEAL,  label="Hist. opt.", ls="--")
(100 * (1 + bench     ).cumprod()).plot(ax=ax, color=GRAY,  label="SPY",        lw=0.9, ls=":")
ax.set_title("Cumulative Return (base=100)"); ax.legend(fontsize=8)
ax.spines[["top","right"]].set_visible(False)

# Drawdown
ax = axes[0, 1]
dd = (1 + port_garch).cumprod()
dd = (dd / dd.cummax() - 1) * 100
ax.fill_between(dd.index, dd, 0, color=CORAL, alpha=0.55)
ax.set_title("Portfolio Drawdown (%)"); ax.spines[["top","right"]].set_visible(False)

# Rolling vol
ax = axes[1, 0]
(port_garch.rolling(21).std() * np.sqrt(252) * 100).plot(ax=ax, color=NAVY, lw=1.2)
ax.set_title("Rolling 21d Volatility (%)"); ax.spines[["top","right"]].set_visible(False)

# Weights bar
ax = axes[1, 1]
w_plot = w_garch[w_garch > 0.005].sort_values()
ax.barh(w_plot.index, w_plot * 100, color=NAVY)
ax.set_title("Portfolio Weights — GARCH opt. (%)"); ax.spines[["top","right"]].set_visible(False)

plt.tight_layout()
plt.savefig("results.png", dpi=150, bbox_inches="tight")
print("\nChart saved → results.png")
