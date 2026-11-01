import numpy as np
import pytest
from scipy.stats import norm
from derivkit import HestonPricer, BlackScholesPricer
from derivkit.contracts import EuropeanCall


S0, r, q, T = 100.0, 0.05, 0.02, 0.5
strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
X = np.column_stack([strikes, np.full(5, T)])
sigma_true = 0.25


def bs_call(K, sigma):
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


@pytest.fixture(scope="module")
def bs():
    return BlackScholesPricer(S0=S0, r=r, q=q).fit(X, bs_call(strikes, sigma_true))


@pytest.fixture(scope="module")
def heston():
    pricer = HestonPricer(S0=S0, r=r, q=q)
    return pricer.fit(X[1:4], [12.0, 6.0, 2.5], np.array([[1.0, 0.3, 0.04, -0.5, 0.04]]))


# ------------------------------------------------------------------
# Shape, seed and validation
# ------------------------------------------------------------------

@pytest.mark.parametrize("name", ["bs", "heston"])
def test_simulate_shape_and_sign(name, request):
    paths = request.getfixturevalue(name).simulate(T=T, steps=10, n_paths=50, seed=0)
    assert paths.shape == (50, 11)
    assert np.all(paths > 0)
    assert np.all(paths[:, 0] == S0)


@pytest.mark.parametrize("name", ["bs", "heston"])
def test_simulate_seed_is_reproducible(name, request):
    pricer = request.getfixturevalue(name)
    np.testing.assert_array_equal(pricer.simulate(T, 10, 50, seed=1), pricer.simulate(T, 10, 50, seed=1))
    assert not np.array_equal(pricer.simulate(T, 10, 50, seed=1), pricer.simulate(T, 10, 50, seed=2))


@pytest.mark.parametrize("kwargs", [dict(T=0, steps=10), dict(T=T, steps=0), dict(T=T, steps=10, n_paths=0)])
def test_simulate_rejects_bad_grid(bs, kwargs):
    with pytest.raises(ValueError):
        bs.simulate(**kwargs)


# ------------------------------------------------------------------
# FFT and simulation describe the same model
# ------------------------------------------------------------------

def test_bs_fit_recovers_sigma_with_dividend_yield(bs):
    np.testing.assert_allclose(bs.params_[0], sigma_true, atol=1e-3)


@pytest.mark.parametrize("name", ["bs", "heston"])
def test_simulated_forward_includes_dividend_yield(name, request):
    paths = request.getfixturevalue(name).simulate(T=T, steps=50, n_paths=100_000, seed=0)
    np.testing.assert_allclose(paths[:, -1].mean(), S0 * np.exp((r - q) * T), rtol=5e-3)


@pytest.mark.parametrize("name", ["bs", "heston"])
def test_fft_and_monte_carlo_agree(name, request):
    pricer = request.getfixturevalue(name)
    fft_price = pricer.predict([[100.0, T]])[0]
    paths = pricer.simulate(T=T, steps=200, n_paths=100_000, seed=0)
    mc_price = EuropeanCall(strike=100.0, T=T, r=r).price(paths)
    np.testing.assert_allclose(mc_price, fft_price, atol=0.1)
