"""
Modèle de Black-Scholes-Merton : prix et Grecques (options européennes vanille).

Convention des paramètres utilisée partout dans le projet :
    S     : prix spot du sous-jacent
    K     : strike (prix d'exercice)
    T     : maturité en années (ex: 0.5 pour 6 mois)
    r     : taux sans risque (continu, ex: 0.03 pour 3%)
    sigma : volatilité annualisée (ex: 0.20 pour 20%)
    q     : taux de dividende continu (0 par défaut)
"""

from __future__ import annotations
import numpy as np
from scipy.stats import norm


def _d1_d2(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0):
    """Calcule d1 et d2, les deux quantités centrales de la formule de BS."""
    if T <= 0 or sigma <= 0:
        raise ValueError("T et sigma doivent être strictement positifs.")
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bs_price(S: float, K: float, T: float, r: float, sigma: float,
             option_type: str = "call", q: float = 0.0) -> float:
    """Prix Black-Scholes d'un call ou d'un put européen."""
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)

    if option_type == "call":
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    else:
        raise ValueError("option_type doit être 'call' ou 'put'.")

    return float(price)


def greeks(S: float, K: float, T: float, r: float, sigma: float,
           option_type: str = "call", q: float = 0.0) -> dict:
    """
    Renvoie les Grecques principales sous forme de dict :
    delta, gamma, vega, theta, rho.

    - delta : sensibilité du prix à S
    - gamma : sensibilité du delta à S (convexité)
    - vega  : sensibilité du prix à sigma (pour 1% de vol, on divise par 100)
    - theta : sensibilité du prix au temps (par jour, on divise par 365)
    - rho   : sensibilité du prix au taux r (pour 1%, on divise par 100)
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    pdf_d1 = norm.pdf(d1)
    disc_q = np.exp(-q * T)
    disc_r = np.exp(-r * T)

    gamma = disc_q * pdf_d1 / (S * sigma * np.sqrt(T))
    vega = S * disc_q * pdf_d1 * np.sqrt(T)

    if option_type == "call":
        delta = disc_q * norm.cdf(d1)
        theta = (-S * disc_q * pdf_d1 * sigma / (2 * np.sqrt(T))
                 - r * K * disc_r * norm.cdf(d2)
                 + q * S * disc_q * norm.cdf(d1))
        rho = K * T * disc_r * norm.cdf(d2)
    elif option_type == "put":
        delta = disc_q * (norm.cdf(d1) - 1)
        theta = (-S * disc_q * pdf_d1 * sigma / (2 * np.sqrt(T))
                 + r * K * disc_r * norm.cdf(-d2)
                 - q * S * disc_q * norm.cdf(-d1))
        rho = -K * T * disc_r * norm.cdf(-d2)
    else:
        raise ValueError("option_type doit être 'call' ou 'put'.")

    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega) / 100,   # pour 1 point de vol
        "theta": float(theta) / 365,  # par jour calendaire
        "rho": float(rho) / 100,      # pour 1 point de taux
    }


if __name__ == "__main__":
    # Petit exemple de sanity check quand on lance le fichier seul
    S, K, T, r, sigma = 100, 100, 1.0, 0.03, 0.20
    print(f"Call price : {bs_price(S, K, T, r, sigma, 'call'):.4f}")
    print(f"Put price  : {bs_price(S, K, T, r, sigma, 'put'):.4f}")
    print("Grecques call:", greeks(S, K, T, r, sigma, "call"))
