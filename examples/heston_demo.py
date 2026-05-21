"""
Full end-to-end demo: Heston calibration, Monte Carlo, and Vixification.
Run from the repo root: python examples/heston_demo.py
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from derivkit import HestonPricer
from derivkit.diagnostics import evaluate, plot_fit, plot_paths

sns.set_theme(context="notebook", style="whitegrid")
np.random.seed(124)

data = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "Data.csv"))
S0 = 100
r = 0.05

prices = data["Prices"].to_numpy()
strikes = data["Strikes"].to_numpy()
maturities = data["Maturities"].to_numpy()

X = np.column_stack([strikes, maturities])
y = prices

# ------------------------------------------------------------------
# Calibration
# ------------------------------------------------------------------

starting_values = np.array([
    [1.0, 0.3, 0.04, -0.5, 0.04],
    [2.0, 0.5, 0.06, -0.7, 0.06],
    [0.5, 0.2, 0.02, -0.3, 0.02],
    [1.5, 0.7, 0.05, -0.6, 0.05],
    [1.0, 0.3, 0.08, -0.5, 0.08],
    [1.0, 0.3, 0.04, -0.9, 0.04],
    [1.0, 0.3, 0.15, -0.5, 0.15],
])  # columns: (kappa, theta, eta, rho, V0)

pricer = HestonPricer(S0=S0, r=r, verbose=1)
pricer.fit(X, y, starting_values)

plot_fit(pricer, X, y)

# ------------------------------------------------------------------
# Approach 1: Analytical fair strike
# Fair strike under Heston: K_var ≈ η + (V0 − η) * (1 − e^(−κT)) / (κT)
# ------------------------------------------------------------------

kappa, theta, eta, rho, V_0 = pricer.params_
unique_maturities = np.unique(maturities)
fair_strikes = (1 - np.exp(-kappa * unique_maturities)) * (V_0 - eta) / (kappa * unique_maturities) + eta
fair_strike_ANAL = fair_strikes[np.isclose(unique_maturities, 1/3)][0]

print(f"\n{'='*40}")
print(f"{'Parameter':<15} | {'Value':>15}")
print(f"{'-'*40}")
for name, val in zip(['kappa', 'theta', 'eta', 'rho', 'V_0'], pricer.params_):
    print(f"{name:<15} | {val:>15.4f}")
print(f"{'-'*40}")
print(f"{'Fair Strike':<15} | {fair_strike_ANAL:>15.4f}")
print(f"{'='*40}")

fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(unique_maturities, fair_strikes, color='black', s=50)
ax.set_xlabel("Maturity")
ax.set_ylabel("Strike")
ax.set_title("Fair value of variance swap — Analytical")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------
# Approach 2: Monte Carlo
# ------------------------------------------------------------------

from derivkit.contracts import VarianceSwap

T = 4/12
paths = pricer.simulate(T=T, steps=round(T * 252), n_paths=10_000)
vs = VarianceSwap(T=T, r=r)
fair_strike_MC = vs.fair_strike(paths)
print(f"\n{'='*40}")
print(f"{'Monte Carlo Fair Strike':<25} | {fair_strike_MC:>10.4f}")
print(f"{'='*40}")

