import inspect
from abc import ABC, abstractmethod
import numpy as np

class BaseContract(ABC):
    """
    Base class for all derivative contracts.

    Subclasses implement price(paths) and optionally extra methods
    (e.g. fair_strike for variance swaps).
    """

    @abstractmethod
    def price(self, paths: np.ndarray):
        """Calculates the price (expected discounted payoff)
        
        Parameters
        ----------
        paths: np.ndarray
            array of (n_paths x T) containing the spot prices
        """

    def __repr__(self):
        sig = inspect.signature(self.__class__.__init__)
        params = {k: getattr(self, k) for k in sig.parameters if k != 'self'}
        param_str = ', '.join(f'{k}={v!r}' for k, v in params.items())
        return f'{self.__class__.__name__}({param_str})'
