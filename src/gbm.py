import numpy as np


def simulate_path(Sof0, mu, sigma, T, N, seed=None):
    """
    Sof0 : float
        Initial stock price
    mu : float
        Annual drift
    sigma : float
        Annual volatility
    T : float
        Time horizon (yr)
    N: int
        Number of time steps
    seed : int, optional
        Random seed

    Returns
    -------
    np.ndarray
        Array of length N+1 containing simulated prices on [0, T]
    """
    rng = np.random.default_rng(seed)
    dt = T / N
    Z = rng.standard_normal(N)
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    path = Sof0 * np.exp(np.cumsum(log_returns))
    return np.concatenate([[Sof0], path])


def simulate_paths(Sof0, mu, sigma, T, N, M, seed=None):
    """
    Sof0 : float
        Initial stock price
    mu : float
        Annual drift
    sigma : float
        Annual volatility
    T : float
        Time horizon (yr)
    N : int
        Number of time steps
    M : int
        Number of paths to simulate
    seed : int, optional
        Random seed

    Returns
    -------
    np.ndarray
        Array of shape (N+1, M) — each column is one simulated path.
    """
    rng = np.random.default_rng(seed)
    dt = T / N
    Z = rng.standard_normal((N, M))
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    paths = Sof0 * np.exp(np.cumsum(log_returns, axis=0))
    return np.concatenate([np.full((1, M), Sof0), paths])