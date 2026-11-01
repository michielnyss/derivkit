import numpy as np

from .base import BasePricer


class HestonPricer(BasePricer):
    """
    Heston (1993) stochastic volatility model.

        dS/S = (r - q) dt + sqrt(V) dW_S
        dV   = kappa (eta - V) dt + theta sqrt(V) dW_V,   d<W_S, W_V> = rho dt

    params_ is [kappa, theta, eta, rho, V0]: mean-reversion speed, volatility
    of variance, long-run variance, correlation, initial variance. Parameters
    and fitted attributes as in BasePricer.
    """

    param_names = ('kappa', 'theta', 'eta', 'rho', 'V0')

    # [kappa, theta, eta, rho, V0] — all satisfy the Feller condition 2κη > θ²
    default_starting_values = np.array([
        [2.0, 0.30, 0.04, -0.50, 0.04],
        [3.0, 0.40, 0.06, -0.70, 0.06],
        [1.5, 0.30, 0.08, -0.30, 0.04],
        [4.0, 0.50, 0.05, -0.60, 0.05],
    ])

    def _is_valid(self, params):
        """Parameter constraints including the Feller condition."""
        kappa, theta, eta, rho, V0 = params
        return (kappa > 0 and theta > 0 and eta > 0
                and -1 < rho < 1 and V0 > 0
                and 2 * kappa * eta > theta ** 2)

    def _char_func(self, u, T, params):
        """Heston characteristic function of log(S_T)."""
        kappa, theta, eta, rho, V0 = params
        d = np.sqrt((rho * theta * 1j * u - kappa) ** 2 + theta ** 2 * (1j * u + u ** 2))
        g = (kappa - rho * theta * 1j * u - d) / (kappa - rho * theta * 1j * u + d)
        C = ((self.r - self.q) * 1j * u * T
             + (kappa * eta / theta ** 2)
             * ((kappa - rho * theta * 1j * u - d) * T
                - 2 * np.log((1 - g * np.exp(-d * T)) / (1 - g))))
        D = ((kappa - rho * theta * 1j * u - d) / theta ** 2
             * (1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T)))
        return np.exp(C + D * V0 + 1j * u * np.log(self.S0))

    def _simulate(self, T, steps, n_paths, rng, params):
        """Euler scheme with a reflected variance."""
        kappa, theta, eta, rho, V0 = params
        dt = T / steps
        sqrt_dt = np.sqrt(dt)

        dW_v = rng.standard_normal(size=(steps, n_paths))
        dW_s = rho * dW_v + np.sqrt(1 - rho ** 2) * rng.standard_normal(size=(steps, n_paths))

        vol = np.empty((steps + 1, n_paths))
        price = np.empty((steps + 1, n_paths))
        vol[0], price[0] = V0, self.S0

        for t in range(steps):
            V_t = vol[t]
            sqrt_Vt = np.sqrt(V_t)
            vol[t + 1] = np.abs(
                V_t + kappa * (eta - V_t) * dt + theta * sqrt_Vt * dW_v[t] * sqrt_dt
            )
            price[t + 1] = price[t] * np.exp(
                (self.r - self.q - 0.5 * V_t) * dt + sqrt_Vt * dW_s[t] * sqrt_dt
            )

        return price.T
