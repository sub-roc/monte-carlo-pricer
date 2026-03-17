import numpy as np

# Single path simulation

def simulate_path(Sof0, mu, sigma, T, N, seed=none):
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
        seed : int,  optional
            Random seed

        *Returns*
        np.ndarray
            Array of length N+1 containing simulated prices on [0, T]
        """
    if seed is not None:
        np.random.seed(seed)

    dt = T/N #days per step

    Z = np.random.standard_normal(N)
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    path = Sof0 * np.exp(np.cumsum(log_returns))
    return np.concatenate([[Sof0], path])