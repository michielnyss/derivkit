import numpy as np
import pytest
from scipy.stats import norm
from derivkit import HestonPricer, BlackScholesPricer, NotFittedError


S0, r, q, T = 100.0, 0.05, 0.02, 0.5
heston_params = dict(kappa=2.0, theta=0.3, eta=0.04, rho=-0.5, V0=0.04)


def bs_call(K, sigma):
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def test_bs_from_params_matches_closed_form():
    pricer = BlackScholesPricer.from_params(S0=S0, r=r, q=q, sigma=0.25)
    strikes = np.array([90.0, 100.0, 110.0])
    X = np.column_stack([strikes, np.full(3, T)])
    np.testing.assert_allclose(pricer.predict(X), bs_call(strikes, 0.25), atol=1e-2)


def test_from_params_can_simulate():
    pricer = HestonPricer.from_params(S0=S0, r=r, **heston_params)
    assert pricer.simulate(T=T, steps=10, n_paths=50, seed=0).shape == (50, 11)


def test_save_and_load_roundtrip():
    fitted = HestonPricer(S0=S0, r=r, q=q).fit(
        np.array([[90.0, T], [100.0, T], [110.0, T]]), [12.0, 6.0, 2.5],
        np.array([[2.0, 0.3, 0.04, -0.5, 0.04]]))
    loaded = HestonPricer.from_params(S0=S0, r=r, q=q, **fitted.get_model_params())
    X = np.array([[95.0, T], [105.0, 1.0]])
    np.testing.assert_array_equal(loaded.predict(X), fitted.predict(X))


def test_from_params_rejects_missing_or_unknown_names():
    with pytest.raises(TypeError, match="missing \\['V0'\\]"):
        HestonPricer.from_params(S0=S0, r=r, kappa=2.0, theta=0.3, eta=0.04, rho=-0.5)
    with pytest.raises(TypeError, match="unknown \\['vol'\\]"):
        BlackScholesPricer.from_params(S0=S0, r=r, sigma=0.2, vol=0.2)


def test_from_params_warns_outside_fit_region():
    with pytest.warns(UserWarning, match="outside the region"):
        HestonPricer.from_params(S0=S0, r=r, **{**heston_params, 'theta': 1.0})  # breaks Feller


def test_get_model_params_not_fitted():
    with pytest.raises(NotFittedError):
        HestonPricer(S0=S0, r=r).get_model_params()
