import time

import numpy as np

from ..base import BasePricer, _check_is_fitted


class BlackScholesPricer(BasePricer):
    """
    Black-Scholes constant-volatility pricer.

    Fitted attributes
    -----------------
    params_    : ndarray [sigma]
    loss_      : float — MSE at optimum
    n_iter_    : int   — Nelder-Mead iterations
    fit_time_  : float — calibration time in seconds

    Parameters
    ----------
    S0 : float
        Current underlying price.
    r : float
        Continuously compounded risk-free rate.
    q : float, default=0.0
        Continuous dividend yield.
    fft_params : list [N, alpha, eta], default=[4096, 1.5, 0.25]
        Carr-Madan FFT hyperparameters.
    verbose : int, default=0
        0 = silent. 1 = print summary after fit.
    """

    def __init__(self, S0, r, q=0.0, fft_params=None, verbose=0):
        self.S0 = S0
        self.r = r
        self.q = q
        self.fft_params = fft_params
        self.verbose = verbose

    def fit(self, X, y, starting_values):
        """
        Calibrate implied volatility to market option prices.

        Parameters
        ----------
        X : array-like (n_samples, 2)
            Columns: [strike, maturity].
        y : array-like (n_samples,)
            Observed market call prices.
        starting_values : array-like (n_starts, 1)
            Initial sigma guesses.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        starting_values = np.atleast_2d(np.asarray(starting_values, dtype=float))

        self.n_features_in_ = 2
        unique_T, grouped_strikes, grouped_prices = self._group_by_maturity(X, y)
        N, alpha, eta = self._resolved_fft_params

        def loss_fn(params):
            sigma, = params
            if sigma <= 0:
                return 1e10
            total = 0.0
            for i, T in enumerate(unique_T):
                predicted = self._fft_pricer(self._char_func(T, params), T, grouped_strikes[i], N, alpha, eta)
                total += float(np.sum((predicted - grouped_prices[i]) ** 2))
            return total

        t0 = time.time()
        self.params_, meta = self._optimize(loss_fn, starting_values)
        self.fit_time_ = time.time() - t0
        self.loss_ = meta['score']
        self.n_iter_ = meta['nit']

        if self.verbose:
            sigma, = self.params_
            print(repr(self))
            print(f"  params=(sigma={sigma:.4f})")
            print(f"  loss={self.loss_:.4f}  n_iter={self.n_iter_}  time={self.fit_time_:.2f}s")

        return self

    def simulate(self, T, steps, n_paths=10_000):
        """
        Simulate GBM (Black-Scholes) price paths.

        Parameters
        ----------
        T : float
        steps : int
        n_paths : int

        Returns
        -------
        paths : ndarray (n_paths, steps + 1)
        """
        _check_is_fitted(self)
        sigma, = self.params_
        dt = T / steps
        dW = np.random.normal(0, np.sqrt(dt), size=(n_paths, steps))
        t = np.linspace(dt, T, steps)
        exponent = (self.r - 0.5 * sigma ** 2) * t + sigma * np.cumsum(dW, axis=1)
        paths = self.S0 * np.exp(exponent)
        return np.hstack((np.full((n_paths, 1), self.S0), paths))

    def _char_func(self, T, params):
        """Black-Scholes characteristic function of the log-price."""
        sigma, = params
        S0, r = self.S0, self.r
        return lambda u: np.exp(
            1j * u * (np.log(S0) + (r - 0.5 * sigma ** 2) * T)
            - 0.5 * sigma ** 2 * u ** 2 * T
        )
