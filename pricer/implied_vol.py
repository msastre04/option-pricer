"""
Calcul de la volatilité implicite : on inverse la formule de Black-Scholes
pour retrouver le sigma qui, injecté dans BS, redonne le prix de marché observé.
"""

from __future__ import annotations
from scipy.optimize import brentq

try:
    from black_scholes import bs_price
except ImportError:
    from pricer.black_scholes import bs_price


def implied_vol(market_price: float, S: float, K: float, T: float, r: float,
                 option_type: str = "call", q: float = 0.0,
                 vol_bounds: tuple = (1e-4, 5.0)) -> float | None:
    """
    Renvoie la volatilité implicite correspondant à market_price, ou None si
    le solveur ne trouve pas de solution dans vol_bounds (prix incohérent
    avec les bornes d'arbitrage, données bruitées, etc.).
    """
    def objective(sigma):
        return bs_price(S, K, T, r, sigma, option_type, q) - market_price

    try:
        return brentq(objective, vol_bounds[0], vol_bounds[1], xtol=1e-6)
    except ValueError:
        # objective ne change pas de signe sur les bornes -> pas de solution valide
        return None


if __name__ == "__main__":
    # On génère un prix avec une vol connue, puis on vérifie qu'on la retrouve
    S, K, T, r, true_sigma = 100, 105, 0.5, 0.03, 0.22
    price = bs_price(S, K, T, r, true_sigma, "call")
    recovered = implied_vol(price, S, K, T, r, "call")
    print(f"Vol injectée : {true_sigma:.4f}  |  Vol retrouvée : {recovered:.4f}")
