import numpy as np

from .base import BasePricer


class BlackScholesPricer(BasePricer):
    """
    Black-Scholes model: geometric Brownian motion with constant volatility.

        dS/S = (r - q) dt + sigma dW

    params_ is [sigma]. Parameters and fitted attributes as in BasePricer.
    """

    param_names = ('sigma',)
    default_starting_values = np.array([[0.10], [0.20], [0.30], [0.40]])

    def _is_valid(self, params):
        sigma, = params
        return sigma > 0

    def _char_func(self, u, T, params):
        """Black-Scholes characteristic function of log(S_T)."""
        sigma, = params
        return np.exp(
            1j * u * (np.log(self.S0) + (self.r - self.q - 0.5 * sigma ** 2) * T)
            - 0.5 * sigma ** 2 * u ** 2 * T
        )

    def _simulate(self, T, steps, n_paths, rng, params):
        """Exact GBM solution."""
        sigma, = params
        dt = T / steps
        dW = rng.normal(0, np.sqrt(dt), size=(n_paths, steps))
        t = np.linspace(dt, T, steps)
        drift = self.r - self.q - 0.5 * sigma ** 2
        paths = self.S0 * np.exp(drift * t + sigma * np.cumsum(dW, axis=1))
        return np.hstack((np.full((n_paths, 1), self.S0), paths))
