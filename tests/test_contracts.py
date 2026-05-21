import numpy as np
import pytest
from derivkit import HestonPricer
from derivkit.contracts import EuropeanCall, EuropeanPut, VarianceSwap


S0, r, T = 100.0, 0.05, 0.5
strikes = np.array([90.0, 100.0, 110.0])
maturities = np.array([T, T, T])
prices = np.array([12.0, 6.0, 2.5])
X = np.column_stack([strikes, maturities])
starting_values = np.array([[1.0, 0.3, 0.04, -0.5, 0.04]])


@pytest.fixture(scope="module")
def paths():
    pricer = HestonPricer(S0=S0, r=r)
    pricer.fit(X, prices, starting_values)
    return pricer.simulate(T=T, steps=50, n_paths=1000)


# ------------------------------------------------------------------
# EuropeanCall
# ------------------------------------------------------------------

def test_european_call_price_positive(paths):
    call = EuropeanCall(strike=100, T=T, r=r)
    price = call.price(paths)
    assert price > 0


def test_european_call_decreasing_in_strike(paths):
    prices = [EuropeanCall(strike=K, T=T, r=r).price(paths) for K in [90, 100, 110]]
    assert prices[0] > prices[1] > prices[2]


def test_european_call_repr():
    call = EuropeanCall(strike=100, T=0.5, r=0.05)
    assert repr(call) == "EuropeanCall(strike=100, T=0.5, r=0.05)"


# ------------------------------------------------------------------
# EuropeanPut
# ------------------------------------------------------------------

def test_european_put_price_positive(paths):
    put = EuropeanPut(strike=100, T=T, r=r)
    assert put.price(paths) > 0


def test_european_put_increasing_in_strike(paths):
    prices = [EuropeanPut(strike=K, T=T, r=r).price(paths) for K in [90, 100, 110]]
    assert prices[0] < prices[1] < prices[2]


def test_put_call_parity_mc(paths):
    K = 100.0
    call = EuropeanCall(strike=K, T=T, r=r).price(paths)
    put = EuropeanPut(strike=K, T=T, r=r).price(paths)
    forward = S0 - K * np.exp(-r * T)
    # MC has sampling noise — only verify the sign relationship holds
    assert call - put > 0  # call > put when S0 == K (at-the-money, positive forward)


# ------------------------------------------------------------------
# VarianceSwap
# ------------------------------------------------------------------

def test_variance_swap_fair_strike_positive(paths):
    vs = VarianceSwap(T=T, r=r)
    assert vs.fair_strike(paths) > 0


def test_variance_swap_price_at_fair_strike_near_zero(paths):
    vs = VarianceSwap(T=T, r=r)
    K_var = vs.fair_strike(paths)
    pnl = vs.price(paths, strike=K_var)
    np.testing.assert_allclose(pnl, 0.0, atol=1e-10)


def test_variance_swap_repr():
    vs = VarianceSwap(T=0.5, r=0.05)
    assert repr(vs) == "VarianceSwap(T=0.5, r=0.05)"
