import numpy as np

from .base import BaseContract


class _EuropeanBase(BaseContract):
    """Shared initialiser for European vanilla options."""

    def __init__(self, strike, T, r):
        self.strike = strike
        self.T = T
        self.r = r


class EuropeanCall(_EuropeanBase):
    """
    European call option: max(S_T - K, 0).

    Parameters
    ----------
    strike : float
    T : float — time to maturity in years
    r : float — continuously compounded risk-free rate
    """

    def price(self, paths):
        """
        Price from simulated paths.

        Parameters
        ----------
        paths : ndarray (n_paths, steps + 1)

        Returns
        -------
        float
        """
        S_T = paths[:, -1]
        discount = np.exp(-self.r * self.T)
        return float(discount * np.maximum(S_T - self.strike, 0).mean())


class EuropeanPut(_EuropeanBase):
    """
    European put option: max(K - S_T, 0).

    Parameters
    ----------
    strike : float
    T : float — time to maturity in years
    r : float — continuously compounded risk-free rate
    """

    def price(self, paths):
        """
        Price from simulated paths.

        Parameters
        ----------
        paths : ndarray (n_paths, steps + 1)

        Returns
        -------
        float
        """
        S_T = paths[:, -1]
        discount = np.exp(-self.r * self.T)
        return float(discount * np.maximum(self.strike - S_T, 0).mean())
