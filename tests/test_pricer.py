import numpy as np
import pytest
from derivkit import HestonPricer, BlackScholesPricer, put_call_parity, NotFittedError


S0, r = 100.0, 0.05
strikes = np.array([90.0, 100.0, 110.0])
maturities = np.array([0.5, 0.5, 0.5])
prices = np.array([12.0, 6.0, 2.5])
X = np.column_stack([strikes, maturities])


# ------------------------------------------------------------------
# put_call_parity
# ------------------------------------------------------------------

def test_put_call_parity_roundtrip():
    calls = np.array([12.0, 6.0, 2.5])
    puts = put_call_parity(S0, r, 0.0, 0.5, strikes, calls=calls)
    recovered = put_call_parity(S0, r, 0.0, 0.5, strikes, puts=puts)
    np.testing.assert_allclose(recovered, calls, atol=1e-10)


def test_put_call_parity_requires_one_leg():
    with pytest.raises(ValueError):
        put_call_parity(S0, r, 0.0, 0.5, strikes)
    with pytest.raises(ValueError):
        put_call_parity(S0, r, 0.0, 0.5, strikes,
                        calls=prices, puts=prices)


# ------------------------------------------------------------------
# HestonPricer
# ------------------------------------------------------------------

@pytest.fixture
def fitted_heston():
    pricer = HestonPricer(S0=S0, r=r)
    starting_values = np.array([[1.0, 0.3, 0.04, -0.5, 0.04]])
    return pricer.fit(X, prices, starting_values)


def test_heston_fit_returns_self():
    pricer = HestonPricer(S0=S0, r=r)
    result = pricer.fit(X, prices, np.array([[1.0, 0.3, 0.04, -0.5, 0.04]]))
    assert result is pricer


def test_heston_fitted_attributes(fitted_heston):
    assert hasattr(fitted_heston, 'params_')
    assert hasattr(fitted_heston, 'loss_')
    assert hasattr(fitted_heston, 'n_iter_')
    assert hasattr(fitted_heston, 'fit_time_')
    assert len(fitted_heston.params_) == 5


def test_heston_predict_shape_and_sign(fitted_heston):
    preds = fitted_heston.predict(X)
    assert preds.shape == (3,)
    assert np.all(preds > 0)
    assert preds[0] > preds[1] > preds[2]  # decreasing in strike


def test_heston_score_is_negative(fitted_heston):
    s = fitted_heston.score(X, prices)
    assert s <= 0


def test_heston_not_fitted_error():
    pricer = HestonPricer(S0=S0, r=r)
    with pytest.raises(NotFittedError):
        pricer.predict(X)
    with pytest.raises(NotFittedError):
        pricer.score(X, prices)
    with pytest.raises(NotFittedError):
        pricer.simulate(0.5, 10)


def test_heston_get_params():
    pricer = HestonPricer(S0=S0, r=r, q=0.01, verbose=1)
    params = pricer.get_params()
    assert params == {'S0': S0, 'r': r, 'q': 0.01, 'fft_params': None, 'verbose': 1}


def test_heston_set_params():
    pricer = HestonPricer(S0=S0, r=r)
    pricer.set_params(r=0.03)
    assert pricer.r == 0.03


def test_heston_repr():
    pricer = HestonPricer(S0=100, r=0.05)
    assert repr(pricer).startswith('HestonPricer(')


def test_heston_predict_put(fitted_heston):
    puts = fitted_heston.predict(X, option_type='put')
    assert puts.shape == (3,)
    assert np.all(puts > 0)
    assert puts[0] < puts[1] < puts[2]  # increasing in strike


def test_heston_predict_put_call_parity(fitted_heston):
    calls = fitted_heston.predict(X, option_type='call')
    puts = fitted_heston.predict(X, option_type='put')
    T = maturities[0]
    expected = S0 * np.exp(-fitted_heston.q * T) - strikes * np.exp(-fitted_heston.r * T)
    np.testing.assert_allclose(calls - puts, expected, rtol=1e-6)


def test_heston_predict_invalid_option_type(fitted_heston):
    with pytest.raises(ValueError):
        fitted_heston.predict(X, option_type='banana')


def test_heston_simulate_shape(fitted_heston):
    paths = fitted_heston.simulate(T=0.5, steps=10, n_paths=50)
    assert paths.shape == (50, 11)
    assert np.all(paths > 0)



# ------------------------------------------------------------------
# BlackScholesPricer
# ------------------------------------------------------------------

@pytest.fixture
def fitted_bs():
    pricer = BlackScholesPricer(S0=S0, r=r)
    return pricer.fit(X, prices, np.array([[0.2]]))


def test_bs_fit_returns_self():
    pricer = BlackScholesPricer(S0=S0, r=r)
    result = pricer.fit(X, prices, np.array([[0.2]]))
    assert result is pricer


def test_bs_fitted_attributes(fitted_bs):
    assert hasattr(fitted_bs, 'params_')
    assert len(fitted_bs.params_) == 1
    assert fitted_bs.params_[0] > 0  # sigma > 0


def test_bs_predict_shape_and_sign(fitted_bs):
    preds = fitted_bs.predict(X)
    assert preds.shape == (3,)
    assert np.all(preds > 0)


def test_bs_not_fitted_error():
    pricer = BlackScholesPricer(S0=S0, r=r)
    with pytest.raises(NotFittedError):
        pricer.predict(X)


def test_bs_simulate_shape(fitted_bs):
    paths = fitted_bs.simulate(T=0.5, steps=10, n_paths=50)
    assert paths.shape == (50, 11)
    assert np.all(paths > 0)


def test_bs_repr():
    pricer = BlackScholesPricer(S0=100, r=0.05)
    assert repr(pricer).startswith('BlackScholesPricer(')
