
"""
Consolidated Volatility Surface Modeling Engine.

Handles Black-Scholes implied total variance calculation,
basic vertical arbitrage checks, and SVI calibration.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm


# ==============================================================================
# SECTION 1: BLACK-SCHOLES OPTION PRICING
# ==============================================================================

def black_scholes_price(S, K, T, v, option_type="C"):
    """
    Calculate Black-Scholes option price.

    Parameters:
        S           : Underlying spot price
        K           : Strike price
        T           : Time to expiry in years
        v           : Total variance (sigma^2 * T)
        option_type : C, P, CALL or PUT

    Returns:
        Theoretical option price.
    """

    option_type = str(option_type).strip().upper()

    if option_type in ("CALL", "C"):
        option_type = "C"

    elif option_type in ("PUT", "P"):
        option_type = "P"

    else:
        raise ValueError(
            f"Unknown option type: {option_type}"
        )

    if not np.isfinite(S) or not np.isfinite(K):
        return np.nan

    if not np.isfinite(T) or not np.isfinite(v):
        return np.nan

    if S <= 0 or K <= 0 or T <= 0 or v <= 0:
        return np.nan

    sqrt_v = np.sqrt(v)

    d1 = (
        np.log(S / K) + 0.5 * v
    ) / sqrt_v

    d2 = d1 - sqrt_v

    if option_type == "C":

        price = (
            S * norm.cdf(d1)
            - K * norm.cdf(d2)
        )

    else:

        price = (
            K * norm.cdf(-d2)
            - S * norm.cdf(-d1)
        )

    return float(price)


# ==============================================================================
# SECTION 2: IMPLIED TOTAL VARIANCE CALCULATION
# ==============================================================================

def extract_implied_variance(
    S,
    K,
    T,
    target_price,
    option_type="C",
    max_iter=100,
    tolerance=1e-6
):
    """
    Calculate implied total variance using Newton-Raphson.

    Total variance = sigma^2 * T.

    Returns:
        Implied total variance or NaN if no solution is found.
    """

    if (
        not np.isfinite(S)
        or not np.isfinite(K)
        or not np.isfinite(T)
        or not np.isfinite(target_price)
    ):
        return np.nan

    if S <= 0 or K <= 0 or T <= 0 or target_price <= 0:
        return np.nan

    option_type = str(option_type).strip().upper()

    if option_type in ("CALL", "C"):
        option_type = "C"

    elif option_type in ("PUT", "P"):
        option_type = "P"

    else:
        return np.nan

    # Initial volatility assumption: 50%
    v_est = 0.25 * T

    for _ in range(max_iter):

        price = black_scholes_price(
            S,
            K,
            T,
            v_est,
            option_type
        )

        if not np.isfinite(price):
            return np.nan

        difference = price - target_price

        if abs(difference) < tolerance:
            return float(v_est)

        sqrt_v = np.sqrt(v_est)

        d1 = (
            np.log(S / K) + 0.5 * v_est
        ) / sqrt_v

        # Derivative of option price with respect to total variance
        vega_v = (
            S * norm.pdf(d1)
            / (2 * sqrt_v)
        )

        if vega_v < 1e-8:
            break

        new_v = v_est - difference / vega_v

        # Keep total variance positive
        if new_v <= 0:
            new_v = v_est / 2

        if new_v > 5:
            break

        v_est = new_v

    return np.nan


# ==============================================================================
# SECTION 3: VERTICAL ARBITRAGE CHECK
# ==============================================================================

def audit_vertical_arbitrage(
    strike_array,
    price_array,
    option_type="C"
):
    """
    Check basic vertical spread relationships.

    Calls:
        Prices should decrease as strike increases.

    Puts:
        Prices should increase as strike increases.

    Returns:
        Boolean mask indicating observations passing the check.
    """

    strikes = np.asarray(strike_array, dtype=float)

    prices = np.asarray(price_array, dtype=float)

    if len(strikes) != len(prices):
        raise ValueError(
            "Strike and price arrays must have equal length."
        )

    option_type = str(option_type).strip().upper()

    if option_type in ("CALL", "C"):
        option_type = "C"

    elif option_type in ("PUT", "P"):
        option_type = "P"

    else:
        raise ValueError(
            f"Unknown option type: {option_type}"
        )

    mask = np.ones(len(strikes), dtype=bool)

    if len(strikes) < 2:
        return mask

    # Sort observations by strike
    order = np.argsort(strikes)

    sorted_strikes = strikes[order]
    sorted_prices = prices[order]

    sorted_mask = np.ones(len(strikes), dtype=bool)

    for i in range(len(strikes)):

        if (
            not np.isfinite(sorted_strikes[i])
            or not np.isfinite(sorted_prices[i])
        ):
            sorted_mask[i] = False

    for i in range(1, len(strikes)):

        if (
            not sorted_mask[i]
            or not sorted_mask[i - 1]
        ):
            continue

        if option_type == "C":

            if sorted_prices[i] > sorted_prices[i - 1]:
                sorted_mask[i] = False
                sorted_mask[i - 1] = False

        else:

            if sorted_prices[i] < sorted_prices[i - 1]:
                sorted_mask[i] = False
                sorted_mask[i - 1] = False

    mask[order] = sorted_mask

    return mask


# ==============================================================================
# SECTION 4: SVI VOLATILITY SURFACE FORMULA
# ==============================================================================

def svi_formula(k, params):
    """
    Calculate total implied variance using the SVI model.

    Formula:

    w(k) = a + b * (
        rho * (k - m)
        + sqrt((k - m)^2 + sigma^2)
    )

    k      : Log-forward moneyness
    params : a, b, rho, m, sigma
    """

    a, b, rho, m, sigma = params

    return (
        a
        + b * (
            rho * (k - m)
            + np.sqrt((k - m) ** 2 + sigma ** 2)
        )
    )


# ==============================================================================
# SECTION 5: SVI CALIBRATION
# ==============================================================================

def calibrate_svi_smile(
    log_moneyness,
    implied_variances
):
    """
    Fit SVI parameters to observed implied total variance.

    Returns:
        Calibrated parameters:
        a, b, rho, m, sigma
    """

    k_arr = np.asarray(
        log_moneyness,
        dtype=float
    )

    v_arr = np.asarray(
        implied_variances,
        dtype=float
    )

    if len(k_arr) != len(v_arr):
        raise ValueError(
            "Moneyness and variance arrays must have equal length."
        )

    # Remove invalid observations
    valid = (
        np.isfinite(k_arr)
        & np.isfinite(v_arr)
        & (v_arr > 0)
    )

    k_arr = k_arr[valid]
    v_arr = v_arr[valid]

    if len(k_arr) < 5:
        return np.full(5, np.nan)

    def objective(params):

        predicted_variance = svi_formula(
            k_arr,
            params
        )

        return np.sum(
            (predicted_variance - v_arr) ** 2
        )

    # Initial parameter estimates
    initial_guess = [
        0.05,
        0.1,
        -0.3,
        0.0,
        0.1
    ]

    # Parameter bounds
    bounds = [
        (1e-5, 2.0),   # a
        (1e-5, 2.0),   # b
        (-0.99, 0.99), # rho
        (-1.0, 1.0),   # m
        (1e-5, 1.0)    # sigma
    ]

    result = minimize(
        objective,
        initial_guess,
        method="L-BFGS-B",
        bounds=bounds
    )

    if result.success:
        return result.x

    return np.full(5, np.nan)
