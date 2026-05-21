# Option Pricer

A Python toolkit for pricing and calibrating derivatives using analytical, Monte Carlo, and model-free methods. Supports Black-Scholes and Heston stochastic volatility models.

---

## Features

- **FFT-based pricing** via Carr-Madan method for fast European option valuation
- **Model calibration** using multi-start Nelder-Mead optimization
- **Monte Carlo simulation** for European options and variance swaps
- **Model-free variance estimation** (VIX replication / vixification)
- **Put-call parity** to derive missing option legs

### Supported Models

| Model | Parameters |
|---|---|
| Black-Scholes | σ (volatility) |
| Heston | κ, θ, η, ρ, V₀ |

---

## Getting Started

### Prerequisites

- Python 3.9+
- pip or conda

### Installation

```bash
git clone https://github.com/<your-username>/Option-Pricer.git
cd Option-Pricer
pip install -r requirements.txt
```

### Data Format

The pricer expects a CSV file with three columns:

```
Strikes, Maturities, Prices
90, 0.25, 12.5
100, 0.25, 5.2
...
```

The included `Data.csv` can be used as a reference or replaced with your own market data.

---

## Usage

```python
import numpy as np
import pandas as pd
from derivkit import HestonPricer

# Load market data
data = pd.read_csv("Data.csv")
X = data[["Strikes", "Maturities"]].to_numpy()
y = data["Prices"].to_numpy()

pricer = HestonPricer(S0=100, r=0.05, q=0.0)

# Calibrate Heston model (n_starts x n_params array of starting points)
starting_values = np.array([[1.0, 0.3, 0.04, -0.5, 0.04]])  # (kappa, theta, eta, rho, V0)
pricer.fit(X, y, starting_values)

# Price calls and puts via FFT using calibrated parameters
X_new = np.array([[95.0, 0.5], [100.0, 0.5], [105.0, 0.5]])
call_prices = pricer.predict(X_new, option_type='call')
put_prices  = pricer.predict(X_new, option_type='put')
```

Run the full example:

```bash
python examples/heston_demo.py
```

---

## Project Structure

```
Option-Pricer/
├── derivkit/          # Package: HestonPricer, BlackScholesPricer, contracts, diagnostics
├── examples/          # Runnable demos
├── tests/             # Unit tests
├── Data.csv           # Sample market data (strikes, maturities, prices)
├── Paper.pdf          # Reference paper with theoretical background
├── pyproject.toml     # Package metadata
├── requirements.txt   # Python dependencies
└── README.md
```

---

## Theoretical Background

The implementation is based on:

- **Carr & Madan (1999)** — FFT option pricing via characteristic functions
- **Heston (1993)** — Stochastic volatility model
- **CBOE VIX methodology** — Model-free variance swap replication

See `Paper.pdf` for a detailed derivation of the methods.


