import numpy as np
from scipy.stats import norm

def bs_pricing(Sof0, K, r, sigma, T, option_type='call'):
    """
    compute the Black-Scholes price for a European option.

    Sof0 : float
        Initial stock price.
    K : float
        Strike price.
    r : float
        Risk-free interest rate (annual).
    sigma : float
        Annual volatility.
    T : float
        Time to expiry in years.
    option_type : str
        'call' or 'put'.

    Returns
    -------
    float
        Black-Scholes option price.
    """
    d1 = (np.log(Sof0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        return Sof0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        return K * np.exp(-r * T) * norm.cdf(-d2) - Sof0 * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be either 'call' or 'put'.")

def mc_pricing(Sof0, K, r, sigma, T, N, M, option_type='call', seed=None):
    """
    price a European option using Monte Carlo simulation.

    Returns
    -------
    float
    Monte Carlo estimate of the option price.
    """
    from gbm import simulate_paths

    paths = simulate_paths(Sof0, r, sigma, T, N, M, seed=seed)
    final_prices = paths[-1, :]

    if option_type == 'call':
        payoffs = np.maximum(final_prices - K, 0)
    elif option_type == 'put':
        payoffs = np.maximum(K - final_prices, 0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return np.exp(-r * T) * payoffs.mean()

def compute_greeks(Sof0, K, r, sigma, T, h=0.01, option_type='call'):
    """
    Compute Black-Scholes Greeks via central finite differences.

    Parameters
    ----------
    Sof0 : float
        Initial stock price.
    K : float
        Strike price.
    r : float
        Risk-free rate.
    sigma : float
        Volatility.
    T : float
        Time to expiry in years.
    h : float
        Bump size for finite differences.
    option_type : str
        'call' or 'put'.

    Returns
    -------
    dict
        Dictionary of Greek name to value.
    """
    price = bs_pricing(Sof0, K, r, sigma, T, option_type)

    delta = (bs_pricing(Sof0+h, K, r, sigma, T, option_type) -
             bs_pricing(Sof0-h, K, r, sigma, T, option_type)) / (2*h)

    gamma = (bs_pricing(Sof0+h, K, r, sigma, T, option_type) -
             2*price +
             bs_pricing(Sof0-h, K, r, sigma, T, option_type)) / (h**2)

    vega  = (bs_pricing(Sof0, K, r, sigma+h, T, option_type) -
             bs_pricing(Sof0, K, r, sigma-h, T, option_type)) / (2*h) / 100

    theta = -(bs_pricing(Sof0, K, r, sigma, T+h, option_type) -
              bs_pricing(Sof0, K, r, sigma, T-h, option_type)) / (2*h) / 365

    rho   = (bs_pricing(Sof0, K, r+h, sigma, T, option_type) -
             bs_pricing(Sof0, K, r-h, sigma, T, option_type)) / (2*h) / 100

    return {
        'price' : round(price, 4),
        'delta' : round(delta, 4),
        'gamma' : round(gamma, 4),
        'vega'  : round(vega,  4),
        'theta' : round(theta, 4),
        'rho'   : round(rho,   4),
    }

from scipy.optimize import brentq


def implied_volatility(market_price, Sof0, K, r, T, option_type='call'):
    """
    Compute the implied volatility of a European option.

    Finds the volatility sigma that makes the Black-Scholes price
    equal to the observed market price, using Brent's root-finding method.

    Parameters
    ----------
    market_price : float
        Observed market price of the option.
    Sof0 : float
        Current stock price.
    K : float
        Strike price.
    r : float
        Risk-free rate.
    T : float
        Time to expiry in years.
    option_type : str
        'call' or 'put'.

    Returns
    -------
    float
        Implied volatility, or np.nan if no solution found.
    """
    # objective: find sigma where BS price - market price = 0
    def objective(sigma):
        return bs_pricing(Sof0, K, r, sigma, T, option_type) - market_price

    # check if a solution exists in [0.001, 10.0]
    try:
        return brentq(objective, 1e-4, 10.0, xtol=1e-6)
    except ValueError:
        return np.nan