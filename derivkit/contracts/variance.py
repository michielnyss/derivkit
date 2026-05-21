import numpy as np

from .base import BaseContract


class VarianceSwap(BaseContract):
    """
    Variance swap contract.

    The fair strike K_var is the break-even fixed leg at which the swap has
    zero present value: PV = e^(-rT) * E[sigma_R^2 - K_var] = 0.

    Parameters
    ----------
    T : float — time to maturity in years
    r : float — continuously compounded risk-free rate
    """

    def __init__(self, T, r):
        self.T = T
        self.r = r

    def fair_strike(self, paths):
        """
        Expected annualised realised variance across paths.
        This is the break-even K_var at which the swap has zero value.

        Parameters
        ----------
        paths : ndarray (n_paths, steps + 1)

        Returns
        -------
        float
        """
        rv = self._realised_variance(paths)
        return float(rv.mean())

    def price(self, paths, strike):
        """
        Discounted P&L of a variance swap given a fixed variance strike.

            PV = e^(-rT) * E[sigma_R^2 - K_var]

        Parameters
        ----------
        paths : ndarray (n_paths, steps + 1)
        strike : float — fixed variance strike K_var

        Returns
        -------
        float
        """
        rv = self._realised_variance(paths)
        discount = np.exp(-self.r * self.T)
        return float(discount * (rv - strike).mean())

    def _realised_variance(self, paths):
        """Annualised realised variance for each path."""
        n = paths.shape[1] - 1
        log_ret = np.log(paths[:, 1:] / paths[:, :-1])
        return (252 / n) * np.sum(log_ret ** 2, axis=1)
