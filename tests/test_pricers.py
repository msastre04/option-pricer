"""
Tests de non-régression : on vérifie que l'arbre binomial et le Monte Carlo
convergent bien vers le prix Black-Scholes (à une tolérance raisonnable près),
et quelques propriétés de bon sens sur les Grecques.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pricer.black_scholes import bs_price, greeks
from pricer.binomial import crr_price
from pricer.monte_carlo import mc_price
from pricer.implied_vol import implied_vol


S, K, T, r, sigma = 100, 100, 1.0, 0.03, 0.20


def test_binomial_converges_to_black_scholes():
    bs = bs_price(S, K, T, r, sigma, "call")
    crr = crr_price(S, K, T, r, sigma, N=1000, option_type="call")
    assert abs(bs - crr) < 0.01


def test_monte_carlo_converges_to_black_scholes():
    bs = bs_price(S, K, T, r, sigma, "call")
    result = mc_price(S, K, T, r, sigma, "call", n_sims=500_000, seed=7)
    # tolérance = 3 écarts-types de l'estimateur (test statistiquement robuste)
    assert abs(bs - result["price"]) < 3 * result["std_error"]


def test_put_call_parity():
    call = bs_price(S, K, T, r, sigma, "call")
    put = bs_price(S, K, T, r, sigma, "put")
    # Parité call-put : C - P = S - K * exp(-rT)
    import numpy as np
    assert abs((call - put) - (S - K * np.exp(-r * T))) < 1e-8


def test_delta_bounds():
    g_call = greeks(S, K, T, r, sigma, "call")
    g_put = greeks(S, K, T, r, sigma, "put")
    assert 0 <= g_call["delta"] <= 1
    assert -1 <= g_put["delta"] <= 0


def test_gamma_identical_for_call_and_put():
    # Le gamma est le même pour un call et un put de mêmes caractéristiques
    g_call = greeks(S, K, T, r, sigma, "call")
    g_put = greeks(S, K, T, r, sigma, "put")
    assert abs(g_call["gamma"] - g_put["gamma"]) < 1e-8


def test_implied_vol_recovers_input_vol():
    price = bs_price(S, K, T, r, sigma, "call")
    recovered = implied_vol(price, S, K, T, r, "call")
    assert abs(recovered - sigma) < 1e-4


def test_american_put_at_least_as_valuable_as_european():
    from pricer.binomial import crr_price
    put_am = crr_price(S, K, T, r, sigma, N=300, option_type="put", american=True)
    put_eu = crr_price(S, K, T, r, sigma, N=300, option_type="put", american=False)
    assert put_am >= put_eu - 1e-8
