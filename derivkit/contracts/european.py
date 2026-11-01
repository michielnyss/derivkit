import numpy as np

from .base import BaseContract


class EuropeanCall(BaseContract):
    """
    European call option: max(S_T - K, 0).

    Parameters
    ----------
    strike : float
    T : float
        time to maturity in years
    r : float = 0
        continuously compounded risk-free rate
    """
    
    def __init__(self, strike: float, T: float, r: float = 0):
        self.strike = strike
        self.T = T
        self.r = r

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


class EuropeanPut(BaseContract):
    """
    European put option: max(K - S_T, 0).

    Parameters
    ----------
    strike : float
    T : float
        time to maturity in years
    r : float = 0
        continuously compounded risk-free rate
    """
    
    def __init__(self, strike: float, T: float, r: float = 0):
        self.strike = strike
        self.T = T
        self.r = r

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
