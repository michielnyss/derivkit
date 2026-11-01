import inspect
import time
import warnings
from abc import ABC, abstractmethod

import numpy as np
from scipy.optimize import minimize

from ..utils import _check_is_fitted, put_call_parity


class BasePricer(ABC):
    """
    Base class for all derivkit pricers.

    One pricer is one model. It calibrates to European call prices with FFT
    (fit) or loads known parameters (from_params), prices European options
    with FFT (predict) and simulates paths for Monte Carlo (simulate). FFT
    and simulation use the same parameters, so their prices agree.

    Subclasses set param_names and default_starting_values, and implement
    _char_func() and _simulate(). Override _is_valid() to constrain the
    parameters during calibration.

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

    Fitted attributes
    -----------------
    params_    : ndarray, in the order of param_names
    loss_      : float — sum of squared errors at optimum
    n_iter_    : int   — Nelder-Mead iterations
    fit_time_  : float — calibration time in seconds
    """

    param_names = ()
    default_starting_values = None

    def __init__(self, S0, r, q=0.0, fft_params=None, verbose=0):
        self.S0 = S0
        self.r = r
        self.q = q
        self.fft_params = fft_params
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Model specific
    # ------------------------------------------------------------------

    @abstractmethod
    def _char_func(self, u, T, params):
        """
        Characteristic function of log(S_T) under the risk-neutral measure.

        Parameters
        ----------
        u : complex ndarray
            Fourier frequencies.
        T : float
            Maturity in years.
        params : ndarray
            Model parameters, in the order of param_names.

        Returns
        -------
        complex ndarray
            phi(u) = E[exp(i u log S_T)], same shape as u.
        """

    @abstractmethod
    def _simulate(self, T, steps, n_paths, rng, params):
        """
        Simulate spot paths under the risk-neutral measure.

        Parameters
        ----------
        T : float
        steps : int
        n_paths : int
        rng : numpy.random.Generator
            The only source of randomness.
        params : ndarray

        Returns
        -------
        paths : ndarray (n_paths, steps + 1)
        """

    def _is_valid(self, params):
        """True if the parameters are admissible. Override to add constraints."""
        return True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def from_params(cls, S0, r, q=0.0, fft_params=None, verbose=0, **params):
        """
        Create a pricer with known model parameters, without calibrating.

        The result can predict and simulate like a fitted pricer. Parameters
        outside the region fit() searches (e.g. Heston without the Feller
        condition) are accepted with a warning.

        Parameters
        ----------
        S0, r, q, fft_params, verbose
            As in the constructor.
        **params : float
            Every name in param_names, e.g. sigma=0.2 for Black-Scholes.

        Returns
        -------
        pricer

        Examples
        --------
        >>> HestonPricer.from_params(S0=100, r=0.05, kappa=2.0, theta=0.3,
        ...                          eta=0.04, rho=-0.5, V0=0.04)
        """
        missing = [k for k in cls.param_names if k not in params]
        unknown = [k for k in params if k not in cls.param_names]
        if missing or unknown:
            raise TypeError(f"{cls.__name__} needs parameters {list(cls.param_names)}; "
                            f"missing {missing}, unknown {unknown}.")
        pricer = cls(S0, r, q, fft_params, verbose)
        values = np.array([params[k] for k in cls.param_names], dtype=float)
        if not pricer._is_valid(values):
            warnings.warn(f"{cls.__name__} parameters {params} are outside the region fit() searches.",
                          stacklevel=2)
        pricer.params_ = values
        return pricer

    def get_model_params(self):
        """
        Return the model parameters as a dict, the inverse of from_params.

        Examples
        --------
        >>> saved = pricer.get_model_params()
        >>> loaded = HestonPricer.from_params(S0=100, r=0.05, **saved)
        """
        _check_is_fitted(self)
        return {k: float(v) for k, v in zip(self.param_names, self.params_)}

    def fit(self, X, y, starting_values=None):
        """
        Calibrate the model parameters to market call prices.

        Parameters
        ----------
        X : array-like (n_samples, 2)
            Columns: [strike, maturity].
        y : array-like (n_samples,)
            Observed market call prices.
        starting_values : array-like (n_starts, n_params), optional
            Initial guesses, columns in the order of param_names.
            Defaults to default_starting_values.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if starting_values is None:
            starting_values = self.default_starting_values
        starting_values = np.atleast_2d(np.asarray(starting_values, dtype=float))

        self.n_features_in_ = 2
        unique_T, grouped_strikes, grouped_prices = self._group_by_maturity(X, y)
        N, alpha, eta = self._resolved_fft_params

        def loss_fn(params):
            if not self._is_valid(params):
                return 1e10
            total = 0.0
            for i, T in enumerate(unique_T):
                predicted = self._fft_pricer(T, params, grouped_strikes[i], N, alpha, eta)
                total += float(np.sum((predicted - grouped_prices[i]) ** 2))
            return total

        t0 = time.time()
        self.params_, meta = self._optimize(loss_fn, starting_values)
        self.fit_time_ = time.time() - t0
        self.loss_ = meta['score']
        self.n_iter_ = meta['nit']

        if self.verbose:
            param_str = ', '.join(f'{k}={v:.4f}' for k, v in zip(self.param_names, self.params_))
            print(repr(self))
            print(f"  params=({param_str})")
            print(f"  loss={self.loss_:.4f}  n_iter={self.n_iter_}  time={self.fit_time_:.2f}s")

        return self

    def predict(self, X, option_type='call'):
        """
        Price European options with FFT using calibrated parameters.

        FFT always prices calls; puts are derived via put-call parity.

        Parameters
        ----------
        X : array-like (n_samples, 2)
            Columns: [strike, maturity].
        option_type : {'call', 'put'}, default='call'

        Returns
        -------
        prices : ndarray (n_samples,)
        """
        _check_is_fitted(self)
        if option_type not in ('call', 'put'):
            raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'.")
        X = np.asarray(X, dtype=float)
        N, alpha, eta = self._resolved_fft_params
        unique_T, idx = np.unique(X[:, 1], return_inverse=True)
        out = np.empty(len(X))
        for i, T in enumerate(unique_T):
            mask = idx == i
            calls = self._fft_pricer(T, self.params_, X[mask, 0], N, alpha, eta)
            if option_type == 'put':
                out[mask] = put_call_parity(self.S0, self.r, self.q, T, X[mask, 0], calls=calls)
            else:
                out[mask] = calls
        return out

    def simulate(self, T, steps, n_paths=10_000, seed=None):
        """
        Simulate spot paths with calibrated parameters, for Monte Carlo.

        Parameters
        ----------
        T : float
            Time horizon in years.
        steps : int
            Number of time steps; the step length is T / steps.
        n_paths : int, default=10_000
            Number of paths.
        seed : int, numpy.random.Generator or None, default=None
            Passed to ``numpy.random.default_rng``. The same seed gives the
            same paths, so several contracts can be priced on common random
            numbers.

        Returns
        -------
        paths : ndarray (n_paths, steps + 1)
            Spot paths, time 0 included.
        """
        _check_is_fitted(self)
        if T <= 0:
            raise ValueError(f"T must be > 0, got {T}")
        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}")
        if n_paths < 1:
            raise ValueError(f"n_paths must be >= 1, got {n_paths}")
        rng = np.random.default_rng(seed)
        return self._simulate(T, steps, n_paths, rng, self.params_)

    def score(self, X, y, metric='rmse'):
        """
        Return negative loss on (X, y). Higher is better (sklearn convention).
        metric: 'rmse' | 'mae' | 'mape'
        """
        y = np.asarray(y, dtype=float)
        residuals = self.predict(X) - y
        if metric == 'rmse':
            return -np.sqrt(np.mean(residuals ** 2))
        if metric == 'mae':
            return -np.mean(np.abs(residuals))
        if metric == 'mape':
            return -np.mean(np.abs(residuals) / y) * 100
        raise ValueError(f"Unknown metric '{metric}'. Choose from 'rmse', 'mae', 'mape'.")

    def get_params(self):
        """Return constructor parameters as a dict."""
        sig = inspect.signature(self.__class__.__init__)
        return {k: getattr(self, k) for k in sig.parameters if k != 'self'}

    def set_params(self, **params):
        """Set constructor parameters."""
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def __repr__(self):
        params = self.get_params()
        param_str = ', '.join(f'{k}={v!r}' for k, v in params.items())
        return f'{self.__class__.__name__}({param_str})'

    # ------------------------------------------------------------------
    # Shared internals
    # ------------------------------------------------------------------

    @property
    def _resolved_fft_params(self):
        return tuple(self.fft_params or [4096, 1.5, 0.25])

    def _fft_pricer(self, T, params, strikes, N, alpha, eta):
        """Carr-Madan FFT pricing for European calls."""
        N = int(N)
        key = (N, alpha, eta)
        if getattr(self, '_fft_cache_key', None) != key:
            lambda_ = (2 * np.pi / N) / eta
            b = lambda_ * N / 2
            j = np.arange(1, N + 1)
            v = eta * (j - 1)
            k = -b + lambda_ * (j - 1)
            sw = (3 + (-1) ** j) / 3
            sw[0] = sw[-1] = 1 / 3
            self._fft_cache_key = key
            self._fft_cache = {
                'u': v - (alpha + 1) * 1j,
                'k': k,
                'sw': sw,
                'exp_ivb': np.exp(1j * v * b),
                'denom': alpha ** 2 + alpha - v ** 2 + 1j * (2 * alpha + 1) * v,
                'price_scale': np.exp(-alpha * k) / np.pi,
            }
        c = self._fft_cache
        phi = self._char_func(c['u'], T, params)
        rho = np.exp(-self.r * T) * phi / c['denom']
        Z = np.real(np.fft.fft(c['exp_ivb'] * rho * eta * c['sw']))
        return np.interp(np.log(strikes), c['k'], c['price_scale'] * Z)

    def _optimize(self, loss_fn, starting_values):
        """Multi-start Nelder-Mead; returns (best_params, meta_dict)."""
        n, m = starting_values.shape
        res_params = np.zeros((n, m))
        res_loss = np.zeros(n)
        meta = []

        for i, x0 in enumerate(starting_values):
            res = minimize(loss_fn, x0, method='Nelder-Mead')
            res_params[i] = res.x
            res_loss[i] = res.fun
            meta.append({'message': res.message, 'nit': res.nit, 'nfev': res.nfev})

        best = int(np.argmin(res_loss))
        return res_params[best], {
            'score': res_loss[best],
            'message': meta[best]['message'],
            'nit': meta[best]['nit'],
            'nfev': meta[best]['nfev'],
        }

    def _group_by_maturity(self, X, y):
        """Split flat (strike, maturity, price) arrays into per-maturity lists."""
        strikes = X[:, 0]
        maturities = X[:, 1]
        unique_T, idx = np.unique(maturities, return_inverse=True)
        grouped_strikes = [strikes[idx == i] for i in range(len(unique_T))]
        grouped_prices = [y[idx == i] for i in range(len(unique_T))]
        return unique_T, grouped_strikes, grouped_prices
