# Monte Carlo options pricer

A European options pricer built using Monte Carlo simulation

- Simulates asset prices via geometric Brownian motion
- Prices European call/put options via Monte Carlo simulation
- Compares Monte Carlo estimates to the Black-Scholes closed form solution
- As simulations increase, the Monte Carlo estimate will converge
- Prices Asian and knock-out barrier options using Monte Carlo simulation
- Pulls market data using `yfinance` to calculate historical volatility, then prices options and compares to market option prices
- Estimates greeks numerically using finite differences, then compares it to results found using Black-Scholes
- Estimates implied volatility using Black-Scholes by determining the value of `sigma` needed for the BS model's price to match the real-life option price

## Background

Built before my first year in Mathematical and Physical Sciences at the University of Toronto. 

## Math

### Geometric Brownian Motion

Stock prices are modelled as a continuous-time stochastic process where
percentage moves (rather than dollar moves) are normally distributed.
The discrete-time update rule is:

$$S_{t+\Delta t} = S_t \cdot \exp\left(\left(\mu - \frac{\sigma^2}{2}\right)\Delta t + \sigma\sqrt{\Delta t}\, Z\right)$$

where:
- $S_t$ — stock price at time $t$
- $\mu$ — annual drift (expected return)
- $\sigma$ — annual volatility
- $\Delta t = T/N$ — length of each time step
- $Z \sim \mathcal{N}(0,1)$ — standard normal random shock

The $\sigma^2/2$ term is the Itô correction, which is a consequence of the
mathematics of continuous compounding that prevents the simulation from
systematically overestimating the average path.

Log returns are normally distributed by construction:

$$\log\left(\frac{S_{t+\Delta t}}{S_t}\right) \sim \mathcal{N}\left(\left(\mu - \frac{\sigma^2}{2}\right)\Delta t,\ \sigma^2 \Delta t\right)$$

### Monte Carlo pricing

Under risk-neutral pricing, the fair value of a derivative is its
expected payoff discounted at the risk-free rate $r$, which replaces the real-world drift $\mu$ in a no-arbitrage market.

The Monte Carlo estimate for a European call is:

$$\hat{P} = e^{-rT} \cdot \frac{1}{M} \sum_{i=1}^{M} \max\!\left(S_T^i - K,\ 0\right)$$

By the law of large numbers, $\hat{P} \to P$ as $M \to \infty$,
where $P$ is the true option price.

### Black-Scholes formula

For a European call, the closed-form Black-Scholes price is:

$$C = S_0 \cdot N(d_1) - K e^{-rT} \cdot N(d_2)$$

$$d_1 = \frac{\ln(S_0/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}}, \qquad d_2 = d_1 - \sigma\sqrt{T}$$

where $\Phi$ is the standard normal CDF. The Monte Carlo estimate
converges to this value as $M \to \infty$, which is how we validate
correctness.

The full derivation requires Itô's lemma and a no-arbitrage argument —
beyond the scope of this project. A self-contained derivation using only
basic probability is given in
[Gibson (2021)](https://ryanagibson.com/posts/simple-black-scholes-derivation/).

### The Greeks

Sensitivities of the option price to each input, estimated via central
finite differences $\partial P / \partial x \approx (P(x+h) - P(x-h)) / 2h$:

| Greek | Symbol | Definition | Intuition |
|-------|--------|-----------|-----------|
| Delta | $\Delta$ | $\partial P / \partial S$ |  change per $1 move in stock |
| Gamma | $\Gamma$ | $\partial^2 P / \partial S^2$ | rate of change of delta |
| Vega  | $\nu$ | $\partial P / \partial \sigma$ | $ change per 1% move in vol |
| Theta | $\Theta$ | $-\partial P / \partial T$ | daily time decay |
| Rho   | $\rho$ | $\partial P / \partial r$ | $ change per 1% move in rates |

## Structure

```
monte-carlo-pricer/
├── notebooks/
│   ├── 01_single_path.ipynb       # Geometric BM simulation, single path
│   ├── 02_many_paths.ipynb        # Fan chart, final price distribution
│   ├── 03_option_pricing.ipynb    # Monte Carlo vs Black-Scholes, convergence
│   ├── 04_interactive.ipynb       # Live interactive visualizer
│   ├── 05_asians_barriers.ipynb   # Pricing Asian and barrier options using Monte Carlo simulation
│   ├── 06_market_data.ipynb       # Pulls market history of a stock using yfinance, estimates volatility, prices options, and compares to market
│   ├── 07_greeks.ipynb            # Estimating greeks numerically, comparing calculations to Monte Carlo results
│   ├── 08_implied_vol.ipynb       # Determines future volatility by determining value of sigma to produce correct option price
├── src/
│   ├── gbm.py                     # simulate_path, simulate_paths
│   ├── pricing.py                 # Monte Carlo and Black-Scholes pricing, numerical Greeks pricing
├── tests/
│   └── test_pricer.py             
└── requirements.txt
```

## How to run

Clone the repo and install dependencies:
```bash
git clone https://github.com/sub-roc/monte-carlo-pricer.git
cd monte-carlo-pricer
pip install -r requirements.txt
```

Then open the notebooks in order:
```bash
jupyter notebook notebooks/01_single_path.ipynb
```

## References

- Hull, J. *Options, Futures, and Other Derivatives*
- [A Simple Derivation of Black-Scholes](https://ryanagibson.com/posts/simple-black-scholes-derivation/)
- [An Intuitive Explanation of Black-Scholes](https://gregorygundersen.com/blog/2024/09/28/black-scholes/)
- [Simulating Geometric Brownian Motion](https://gregorygundersen.com/blog/2024/04/13/simulating-gbm/)