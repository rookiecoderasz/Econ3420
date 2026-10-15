"""Student calculation functions for Assignment One.

Ask AI to complete these five functions using the assignment's formulas.
Keep data downloads, file reading and plotting outside this file.
Inputs and outputs use decimal returns, not percentages. Do not hard-code
tickers, weights, a student's preference, or answers from the practice data.
The report imports this module.
"""

import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def estimate_index_model(excess_returns, stocks, benchmark):
    """Question 2: estimate X_{i,t} = alpha_i + beta_i X_{M,t} + e_{i,t}.

    Inputs: monthly decimal excess returns in a DataFrame; five stock names
    and a distinct benchmark column name. Reject missing/nonfinite inputs.
    Return (estimates, residuals):
      estimates: DataFrame indexed by stocks, columns alpha, beta, r_squared.
                 These code column names store alpha_hat_i and beta_hat_i.
      residuals: DataFrame with the input dates and stock columns in order.
    With an intercept, e_hat_{i,t} = X_{i,t} - alpha_hat_i - beta_hat_i X_{M,t}.
    """
    stocks = list(stocks)
    if not isinstance(excess_returns, pd.DataFrame):
        raise ValueError("excess_returns must be a DataFrame.")
    if len(stocks) != 5 or len(set(stocks)) != 5:
        raise ValueError("Supply five distinct stock names.")
    if benchmark in stocks:
        raise ValueError("The benchmark must not be one of the stocks.")
    missing = [c for c in stocks + [benchmark] if c not in excess_returns.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    data = excess_returns[stocks + [benchmark]].astype(float)
    if len(data) < 3:
        raise ValueError("At least three monthly observations are required.")
    if not np.isfinite(data.to_numpy()).all():
        raise ValueError("Excess returns must be finite with no missing values.")
    x = data[benchmark].to_numpy()
    if np.ptp(x) == 0:
        raise ValueError("The benchmark has no variation.")
    y = data[stocks].to_numpy()
    design = np.column_stack([np.ones(len(x)), x])
    coef = np.linalg.lstsq(design, y, rcond=None)[0]
    resid = y - design @ coef
    sst = ((y - y.mean(axis=0)) ** 2).sum(axis=0)
    sse = (resid ** 2).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = 1.0 - sse / sst
    estimates = pd.DataFrame({"alpha": coef[0], "beta": coef[1], "r_squared": r2},
                             index=pd.Index(stocks))
    residuals = pd.DataFrame(resid, index=data.index, columns=stocks)
    if not np.isfinite(estimates.to_numpy()).all():
        raise ValueError("Regression produced nonfinite estimates.")
    return estimates, residuals


def single_index_covariance(betas, market_excess, residuals):
    """Question 3: construct the monthly single-index covariance table.

    Inputs: beta Series indexed by stocks, market excess-return Series, and
    residual DataFrame from Question 2, all with aligned labels/dates.
    Follow Part 3: sigma_i^2 = beta_i^2 sigma_M^2 + sigma^2(e_i);
    Cov(X_i, X_j) = beta_i beta_j sigma_M^2 for different stocks.
    Substitute estimated betas. Estimate sigma_M^2 with ddof=1, but estimate
    sigma^2(e_i) as sum(e_hat_i,t ** 2) / (T - 2), where T is the month count.
    Require T > 2; the residual denominator allows for intercept and slope.
    Return a DataFrame with stock names on both axes, in beta index order.
    Check symmetry, finite entries and nonnegative diagonal variances.
    """
    names = list(betas.index)
    if len(set(names)) != len(names):
        raise ValueError("Beta labels must be unique.")
    if not set(names).issubset(residuals.columns):
        raise ValueError("Residual columns must include every beta label.")
    if not residuals.index.equals(market_excess.index):
        raise ValueError("Residual and market dates must match.")
    res = residuals[names].to_numpy(dtype=float)
    market = market_excess.to_numpy(dtype=float)
    b = betas.to_numpy(dtype=float)
    t = len(market)
    if t <= 2:
        raise ValueError("At least three months are required.")
    if not (np.isfinite(res).all() and np.isfinite(market).all() and np.isfinite(b).all()):
        raise ValueError("Inputs must be finite.")
    var_m = market_excess.astype(float).var(ddof=1)
    resid_var = (res ** 2).sum(axis=0) / (t - 2)
    cov = np.outer(b, b) * var_m
    cov[np.diag_indices_from(cov)] = b ** 2 * var_m + resid_var
    if not np.isfinite(cov).all() or not np.allclose(cov, cov.T, rtol=1e-12, atol=1e-18):
        raise ValueError("Covariance must be finite and symmetric.")
    if (np.diag(cov) < 0).any():
        raise ValueError("Variances must be nonnegative.")
    return pd.DataFrame(cov, index=names, columns=names)


def optimize_risky_portfolio(annual_excess_means, annual_covariance):
    """Question 4: maximize Sharpe with no negative stock weights.

    Inputs: annual decimal excess means (Series) and annual covariance
    (DataFrame), labelled by the same five stocks. Align labels explicitly.
    SR_P = (E(R_P) - R_f) / sigma_P; the input means are already excess returns.
    Try multiple starting points; check solver success, positive variance,
    nonnegative weights and sum=1. Do not silently accept failed optimization.
    Return a dict with:
      weights: Series indexed by stocks, in input order;
      excess_return, volatility, sharpe: annual statistics as floats.
    Return the estimated risky portfolio even if its premium is nonpositive;
    explain this historical result separately. Questions 5 and 6 use
    these same estimates. A nonpositive premium leads to an all-safe outcome.
    """
    names = list(annual_excess_means.index)
    n = len(names)
    if n == 0 or len(set(names)) != n:
        raise ValueError("Means need unique labels.")
    if set(names) != set(annual_covariance.index) or set(names) != set(annual_covariance.columns):
        raise ValueError("Mean and covariance labels differ.")
    mu = annual_excess_means.to_numpy(dtype=float)
    cov = annual_covariance.loc[names, names].to_numpy(dtype=float)
    if not (np.isfinite(mu).all() and np.isfinite(cov).all()):
        raise ValueError("Inputs must be finite.")
    if not np.allclose(cov, cov.T, rtol=1e-9, atol=1e-14):
        raise ValueError("Covariance must be symmetric.")
    cov = (cov + cov.T) / 2

    def neg_sharpe(w):
        var = w @ cov @ w
        return -(w @ mu) / math.sqrt(max(var, 1e-300))

    rng = np.random.default_rng(12345)
    starts = [np.full(n, 1.0 / n)] + [np.eye(n)[i] for i in range(n)]
    starts += [rng.dirichlet(np.ones(n)) for _ in range(5)]
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    best = None
    for start in starts:
        res = minimize(neg_sharpe, start, method="SLSQP", bounds=[(0.0, 1.0)] * n,
                       constraints=constraints, options={"maxiter": 1000, "ftol": 1e-14})
        if not res.success or not np.isfinite(res.fun):
            continue
        w = np.where(res.x < 1e-10, 0.0, res.x)
        if w.sum() <= 0:
            continue
        w = w / w.sum()
        if w @ cov @ w <= 0:
            continue
        value = neg_sharpe(w)
        if best is None or value < best[0]:
            best = (value, w)
    if best is None:
        raise RuntimeError("Portfolio optimization failed from every starting point.")
    w = best[1]
    variance = float(w @ cov @ w)
    vol = math.sqrt(variance)
    excess = float(w @ mu)
    return {"weights": pd.Series(w, index=names), "excess_return": excess,
            "volatility": vol, "sharpe": excess / vol}


def infer_risk_aversion(trials):
    """Question 5: infer gamma from one CE choice for the selected Q4 portfolio.

    Input: the one-row DataFrame validated by risk_aversion_quiz.load_preferences.
    It contains the choice, full-precision mean_return, volatility,
    certainty_equivalent and certainty_equivalent_wealth, with saved portfolio
    inputs and provenance. Reject missing/invalid choices or mismatched values.
    Part 2: gamma = 2 * (mu_P - c) / sigma_P^2. The menu is calibrated from
    G = annual_premium / volatility^2 and candidate multipliers above one.
    Require a positive annual_premium; no numeric gamma is elicited otherwise.
    Return {'estimates': validated one-row DataFrame with recomputed gamma,
            'gamma': float}. There is no median for a single answer.
    You may reuse risk_aversion_quiz.summarize_trials for validation and
    numerically stable inference from full-precision portfolio inputs.
    The offered choices give a coarse estimate with gamma above G (at most 5G).
    """
    from risk_aversion_quiz import summarize_trials

    summary = summarize_trials(trials)
    row = summary["estimates"].iloc[0]
    premium = float(row["annual_premium"])
    variance = float(row["volatility"]) ** 2
    if not premium > 0:
        raise ValueError("A positive annual premium is required to infer gamma.")
    gamma = float(summary["gamma"])
    ce_gamma = 2.0 * (float(row["mean_return"]) - float(row["certainty_equivalent"])) / variance
    if not math.isfinite(gamma) or gamma <= 0:
        raise ValueError("Gamma must be positive and finite.")
    if not math.isclose(gamma, ce_gamma, rel_tol=1e-6, abs_tol=1e-9):
        raise ValueError("Gamma disagrees with 2*(mu_P - c)/sigma_P^2.")
    return {"estimates": summary["estimates"], "gamma": gamma}


def allocate_portfolio(weights, annual_premium, annual_volatility,
                       gamma, annual_safe_rate, wealth):
    """Question 6: mix the selected Q4 portfolio with the safe asset.

    Inputs: five nonnegative stock weights (Series) summing to one; annual
    decimal premium, volatility and safe rate; gamma; positive wealth.
    Require finite premium/safe rate and positive finite volatility and wealth.
    Use the same Q4 premium and volatility that calibrated the Q5 menu.
    For premium > 0, require positive finite gamma and calculate
    risky_share = annual_premium / (gamma * annual_volatility^2).
    The calibrated Q5 choice automatically gives an interior share. Do not
    clip it. Reject an inconsistent gamma producing a share outside (0,1).
    For premium <= 0, allow gamma=None and return risky_share=0: the
    no-borrowing optimum is all safe. Numeric gamma is not elicited in this case.
    Final portfolio F: E(R_F) = annual_safe_rate + risky_share * annual_premium;
    sigma_F = risky_share * annual_volatility.
    The half-premium test keeps the original weights, volatility and gamma.
    Return a dict with:
      risky_share: formula's risky fraction, or 0 in the exception;
      expected_return, volatility: annual total return and volatility;
      holdings: DataFrame indexed by the five stocks then 'Safe asset',
                columns weight (decimal) and amount (currency units).
    Check that holdings weights sum to one and amounts sum to wealth.
    """
    w = weights.astype(float)
    names = list(w.index)
    if (w.to_numpy() < 0).any() or not np.isfinite(w.to_numpy()).all() \
            or not math.isclose(float(w.sum()), 1.0, rel_tol=0, abs_tol=1e-9):
        raise ValueError("Weights must be finite, nonnegative and sum to one.")
    for name, value in [("annual_premium", annual_premium), ("annual_safe_rate", annual_safe_rate)]:
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite.")
    if not (math.isfinite(annual_volatility) and annual_volatility > 0):
        raise ValueError("Volatility must be positive and finite.")
    if not (math.isfinite(wealth) and wealth > 0):
        raise ValueError("Wealth must be positive and finite.")
    if annual_premium > 0:
        if gamma is None or not (math.isfinite(gamma) and gamma > 0):
            raise ValueError("A positive finite gamma is required.")
        share = annual_premium / (gamma * annual_volatility ** 2)
        if not 0 < share < 1:
            raise ValueError("Implied risky share lies outside (0, 1); gamma is inconsistent.")
    else:
        share = 0.0
    holdings = pd.DataFrame(
        {"weight": np.append(share * w.to_numpy(), 1.0 - share)},
        index=names + ["Safe asset"])
    holdings["amount"] = holdings["weight"] * wealth
    if not math.isclose(holdings["weight"].sum(), 1.0, abs_tol=1e-9) \
            or not math.isclose(holdings["amount"].sum(), wealth, rel_tol=1e-9):
        raise ValueError("Holdings do not sum to the budget.")
    return {"risky_share": float(share),
            "expected_return": float(annual_safe_rate + share * annual_premium),
            "volatility": float(share * annual_volatility),
            "holdings": holdings}
