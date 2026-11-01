import numpy as np


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
