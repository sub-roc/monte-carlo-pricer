import sys
sys.path.append('../src')

import numpy as np
import pytest
from pricer import bs_pricing, mc_pricing


# ── Black-Scholes tests ───────────────────────────────────────────────────────

def test_black_scholes_call_known_value():
    """
    Compare against a known Black-Scholes value.
    S0=100, K=100, r=0.05, sigma=0.20, T=1 → ~$10.45
    This is a well-known textbook result.
    """
    price = bs_pricing(100, 100, 0.05, 0.20, 1.0, option_type='call')
    assert abs(price - 10.45) < 0.05


def test_black_scholes_put_known_value():
    """
    Put equivalent of the above via put-call parity.
    P = C - S0 + K*exp(-rT)
    """
    call = bs_pricing(100, 100, 0.05, 0.20, 1.0, option_type='call')
    put  = bs_pricing(100, 100, 0.05, 0.20, 1.0, option_type='put')
    parity = call - 100 + 100 * np.exp(-0.05 * 1.0)
    assert abs(put - parity) < 1e-10


def test_black_scholes_put_call_parity():
    """
    Put-call parity must hold for any valid set of parameters:
    C - P = S0 - K*exp(-rT)
    """
    Sof0, K, r, sigma, T = 110, 100, 0.03, 0.25, 2.0
    call = bs_pricing(Sof0, K, r, sigma, T, option_type='call')
    put  = bs_pricing(Sof0, K, r, sigma, T, option_type='put')
    lhs  = call - put
    rhs  = Sof0 - K * np.exp(-r * T)
    assert abs(lhs - rhs) < 1e-10


def test_black_scholes_call_positive():
    """Option prices must always be non-negative."""
    price = bs_pricing(100, 105, 0.05, 0.20, 1.0, option_type='call')
    assert price > 0


def test_black_scholes_deep_itm_call():
    """
    A deep in-the-money call (S0 >> K) should be worth approximately
    S0 - K*exp(-rT) — almost all intrinsic value.
    """
    Sof0, K, r, T = 200, 50, 0.05, 1.0
    price     = bs_pricing(Sof0, K, r, 0.20, T, option_type='call')
    intrinsic = Sof0 - K * np.exp(-r * T)
    assert abs(price - intrinsic) < 2.0


def test_black_scholes_invalid_option_type():
    """Should raise ValueError for an unrecognised option type."""
    with pytest.raises(ValueError):
        bs_pricing(100, 100, 0.05, 0.20, 1.0, option_type='banana')


def test_black_scholes_higher_vol_raises_price():
    """
    Higher volatility should always increase option price —
    more uncertainty = more valuable optionality.
    """
    low_vol  = bs_pricing(100, 100, 0.05, 0.10, 1.0, option_type='call')
    high_vol = bs_pricing(100, 100, 0.05, 0.40, 1.0, option_type='call')
    assert high_vol > low_vol


def test_black_scholes_longer_expiry_raises_price():
    """Longer time to expiry should increase option price."""
    short = bs_pricing(100, 100, 0.05, 0.20, 0.5, option_type='call')
    long  = bs_pricing(100, 100, 0.05, 0.20, 2.0, option_type='call')
    assert long > short


# ── Monte Carlo tests ─────────────────────────────────────────────────────────

def test_monte_carlo_close_to_bs_pricing():
    """
    With enough paths, the MC estimate should be within 2% of
    the Black-Scholes price.
    """
    Sof0, K, r, sigma, T = 100, 100, 0.05, 0.20, 1.0
    mc = mc_pricing(Sof0, K, r, sigma, T, N=252, M=50000, seed=42)
    bs = bs_pricing(Sof0, K, r, sigma, T, option_type='call')
    assert abs(mc - bs) / bs < 0.02


def test_monte_carlo_positive():
    """Monte Carlo price must always be non-negative."""
    price = mc_pricing(100, 105, 0.05, 0.20, 1.0, N=252, M=1000, seed=42)
    assert price >= 0


def test_monte_carlo_put_call_parity():
    """
    Put-call parity should hold approximately for Monte Carlo prices
    with a large number of simulations.
    """
    Sof0, K, r, sigma, T = 100, 100, 0.05, 0.20, 1.0
    call = mc_pricing(Sof0, K, r, sigma, T, N=252, M=50000, seed=42, option_type='call')
    put  = mc_pricing(Sof0, K, r, sigma, T, N=252, M=50000, seed=42, option_type='put')
    lhs  = call - put
    rhs  = Sof0 - K * np.exp(-r * T)
    assert abs(lhs - rhs) < 0.50


def test_monte_carlo_reproducible():
    """Same seed should always produce the same price."""
    price1 = mc_pricing(100, 105, 0.05, 0.20, 1.0, N=252, M=1000, seed=99)
    price2 = mc_pricing(100, 105, 0.05, 0.20, 1.0, N=252, M=1000, seed=99)
    assert price1 == price2


def test_monte_carlo_invalid_option_type():
    """Should raise ValueError for an unrecognised option type."""
    with pytest.raises(ValueError):
        mc_pricing(100, 105, 0.05, 0.20, 1.0, N=252, M=1000, option_type='banana')