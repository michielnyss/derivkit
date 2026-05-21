import inspect
import time

import numpy as np
from scipy.optimize import minimize

class NotFittedError(ValueError, AttributeError):
    """Raised when predict/score/simulate is called before fit."""


def _check_is_fitted(estimator):
    if not hasattr(estimator, 'params_'):
        raise NotFittedError(
            f"This {type(estimator).__name__} instance is not fitted yet. "
            "Call 'fit' before using this estimator."
        )


def put_call_parity(S0, r, q, T, strikes, calls=None, puts=None):
    """
    Derive the missing leg of a European option pair via put-call parity.

        C - P = S0*e^(-qT) - K*e^(-rT)

    Provide exactly one of calls or puts.
    """
    disc_S = S0 * np.exp(-q * T)
    disc_K = np.asarray(strikes, dtype=float) * np.exp(-r * T)
    if calls is not None and puts is None:
        return np.asarray(calls, dtype=float) - disc_S + disc_K
    if puts is not None and calls is None:
        return np.asarray(puts, dtype=float) + disc_S - disc_K
    raise ValueError("Provide exactly one of calls or puts.")


class BasePricer:
    """
    Base class for all derivkit pricers.

    Subclasses must implement fit(), predict(), simulate(), and _char_func().
    Everything else — FFT engine, optimizer, Monte Carlo payoffs, plotting,
    vixification — lives here and is shared across models.
    """

    def fit(self, X, y, starting_values):
        raise NotImplementedError

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

    def _fft_pricer(self, char_func, T, strikes, N, alpha, eta):
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
        phi = char_func(c['u'])
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

    def predict(self, X, option_type='call'):
        """
        Price European options using calibrated parameters.

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
            calls = self._fft_pricer(self._char_func(T, self.params_), T, X[mask, 0], N, alpha, eta)
            if option_type == 'put':
                out[mask] = put_call_parity(self.S0, self.r, self.q, T, X[mask, 0], calls=calls)
            else:
                out[mask] = calls
        return out

    def _group_by_maturity(self, X, y):
        """Split flat (strike, maturity, price) arrays into per-maturity lists."""
        strikes = X[:, 0]
        maturities = X[:, 1]
        unique_T, idx = np.unique(maturities, return_inverse=True)
        grouped_strikes = [strikes[idx == i] for i in range(len(unique_T))]
        grouped_prices = [y[idx == i] for i in range(len(unique_T))]
        return unique_T, grouped_strikes, grouped_prices

