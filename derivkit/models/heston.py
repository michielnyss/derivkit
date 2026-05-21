import time

import numpy as np

from ..base import BasePricer, _check_is_fitted


class HestonPricer(BasePricer):
    """
    Heston stochastic volatility model pricer.

    Fitted attributes
    -----------------
    params_    : ndarray [kappa, theta, eta, rho, V0]
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

    # [kappa, theta, eta, rho, V0] — all satisfy the Feller condition 2κη > θ²
    _DEFAULT_STARTING_VALUES = np.array([
        [2.0, 0.30, 0.04, -0.50, 0.04],
        [3.0, 0.40, 0.06, -0.70, 0.06],
        [1.5, 0.30, 0.08, -0.30, 0.04],
        [4.0, 0.50, 0.05, -0.60, 0.05],
    ])

    def fit(self, X, y, starting_values=None):
        """
        Calibrate Heston parameters to market option prices.

        Parameters
        ----------
        X : array-like (n_samples, 2)
            Columns: [strike, maturity].
        y : array-like (n_samples,)
            Observed market call prices.
        starting_values : array-like (n_starts, 5), optional
            Initial guesses for [kappa, theta, eta, rho, V0].
            Defaults to a built-in grid covering typical equity vol regimes.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if starting_values is None:
            starting_values = self._DEFAULT_STARTING_VALUES
        starting_values = np.atleast_2d(np.asarray(starting_values, dtype=float))

        self.n_features_in_ = 2
        unique_T, grouped_strikes, grouped_prices = self._group_by_maturity(X, y)
        N, alpha, eta = self._resolved_fft_params

        def loss_fn(params):
            if not self._is_valid(params):
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
            kappa, theta, eta_p, rho, V0 = self.params_
            print(repr(self))
            print(f"  params=(kappa={kappa:.3f}, theta={theta:.3f}, eta={eta_p:.3f}, "
                  f"rho={rho:.3f}, V0={V0:.3f})")
            print(f"  loss={self.loss_:.4f}  n_iter={self.n_iter_}  time={self.fit_time_:.2f}s")

        return self

    def simulate(self, T, steps, n_paths=10_000):
        """
        Simulate Heston price paths.

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
        kappa, theta, eta, rho, V0 = self.params_
        dt = T / steps
        sqrt_dt = np.sqrt(dt)

        dW_v = np.random.normal(size=(steps, n_paths))
        dW_s = rho * dW_v + np.sqrt(1 - rho ** 2) * np.random.normal(size=(steps, n_paths))

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
                (self.r - 0.5 * V_t) * dt + sqrt_Vt * dW_s[t] * sqrt_dt
            )

        return price.T

    def _char_func(self, T, params):
        """Heston characteristic function of the log-price."""
        kappa, theta, eta, rho, V_0 = params
        S0, r = self.S0, self.r

        def phi(u):
            d = np.sqrt((rho * theta * 1j * u - kappa) ** 2 + theta ** 2 * (1j * u + u ** 2))
            g = (kappa - rho * theta * 1j * u - d) / (kappa - rho * theta * 1j * u + d)
            C = (r * 1j * u * T
                 + (kappa * eta / theta ** 2)
                 * ((kappa - rho * theta * 1j * u - d) * T
                    - 2 * np.log((1 - g * np.exp(-d * T)) / (1 - g))))
            D = ((kappa - rho * theta * 1j * u - d) / theta ** 2
                 * (1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T)))
            return np.exp(C + D * V_0 + 1j * u * np.log(S0))

        return phi

    @staticmethod
    def _is_valid(params):
        """Check Heston parameter constraints including the Feller condition."""
        kappa, theta, eta, rho, V_0 = params
        return (kappa > 0 and theta > 0 and eta > 0
                and -1 < rho < 1 and V_0 > 0
                and 2 * kappa * eta > theta ** 2)
