# derivkit

A Python toolkit for pricing and calibrating derivatives using FFT, Monte Carlo, and model-free methods. Supports Black-Scholes and Heston stochastic volatility models.

---

## Features

- **FFT-based pricing** via Carr-Madan for fast European option valuation
- **Model calibration** using multi-start Nelder-Mead optimization
- **Monte Carlo simulation** for European options and variance swaps
- **Model-free variance estimation** (VIX replication)
- **Put-call parity** to derive missing option legs

### Supported Models

| Model | Parameters |
|---|---|
| Black-Scholes | σ (volatility) |
| Heston | κ, θ, η, ρ, V₀ |

---

## Installation

```bash
git clone https://github.com/michielnyss/derivkit.git
cd derivkit
pip install -e .
```

Requires Python 3.9+.

---

## Usage

### Calibration

```python
import numpy as np
import pandas as pd
from derivkit import HestonPricer, BlackScholesPricer

data = pd.read_csv("Data.csv")
X = data[["Strikes", "Maturities"]].to_numpy()
y = data["Prices"].to_numpy()

# Heston — starting_values optional, built-in defaults used if omitted
pricer = HestonPricer(S0=100, r=0.05, verbose=1)
pricer.fit(X, y)

# Black-Scholes
bs = BlackScholesPricer(S0=100, r=0.05)
bs.fit(X, y)
```

Already have parameters? Load them instead of calibrating:

```python
pricer = HestonPricer.from_params(S0=100, r=0.05, kappa=2.0, theta=0.3,
                                  eta=0.04, rho=-0.5, V0=0.04)

saved = fitted.get_model_params()          # {'kappa': ..., 'theta': ..., ...}
loaded = HestonPricer.from_params(S0=100, r=0.05, **saved)
```

Each pricer is one model that calibrates (`fit`), prices European options with FFT (`predict`) and simulates paths for Monte Carlo (`simulate`). FFT and simulation use the same parameters, so their prices agree.

A new model is one subclass of `BasePricer` with `param_names`, `default_starting_values`, `_char_func()` and `_simulate()`; calibration and pricing come for free.

Pass your own starting grid if you want tighter control:

```python
starting_values = np.array([
    [2.0, 0.3, 0.04, -0.5, 0.04],  # [kappa, theta, eta, rho, V0]
    [3.0, 0.4, 0.06, -0.7, 0.06],
])
pricer.fit(X, y, starting_values)
```

### Pricing

```python
X_new = np.array([[95.0, 0.5], [100.0, 0.5], [105.0, 0.5]])  # [strike, maturity]

calls = pricer.predict(X_new, option_type='call')
puts  = pricer.predict(X_new, option_type='put')
```

### Diagnostics

```python
from derivkit import evaluate, plot_fit, plot_paths

metrics = evaluate(pricer, X, y, verbose=True)
# {'SSE': ..., 'MSE': ..., 'RMSE': ..., 'NRMSE': ...}

plot_fit(pricer, X, y)          # model vs market prices per maturity
plot_paths(paths, T=1)          # simulated price paths, see Monte Carlo below
```

### Monte Carlo with Contracts

Pricers simulate paths; contracts price them. Any contract works on paths from any model, and the same `seed` gives the same paths.

```python
from derivkit.contracts import EuropeanCall, EuropeanPut, VarianceSwap

T = 4 / 12
paths = pricer.simulate(T=T, steps=round(T * 252), n_paths=10_000, seed=42)

call = EuropeanCall(strike=100, T=T, r=0.05)
put  = EuropeanPut(strike=100, T=T, r=0.05)
print(call.price(paths), put.price(paths))

vs = VarianceSwap(T=T, r=0.05)
print(vs.fair_strike(paths))
```

### Utilities

```python
from derivkit import put_call_parity

# Derive put prices from calls (or vice versa)
puts = put_call_parity(S0=100, r=0.05, q=0.0, T=0.5,
                       strikes=[95, 100, 105], calls=calls)
```

---

## Project Structure

```
derivkit/
├── derivkit/          # Package: pricers, contracts, diagnostics
│   ├── models/        # BasePricer (fit, predict, simulate), HestonPricer, BlackScholesPricer
│   ├── contracts/     # BaseContract, EuropeanCall, EuropeanPut, VarianceSwap
│   ├── utils.py       # put_call_parity, NotFittedError
│   └── diagnostics.py # evaluate, plot_fit, plot_paths
├── examples/          # heston_demo.py
├── tests/             # Unit tests
├── Data.csv           # Sample market data
├── Paper.pdf          # Theoretical background
└── pyproject.toml     # Package metadata
```

---

## Theoretical Background

- **Carr & Madan (1999)** — FFT option pricing via characteristic functions
- **Heston (1993)** — Stochastic volatility model
- **CBOE VIX methodology** — Model-free variance swap replication

See `Paper.pdf` for detailed derivations.
