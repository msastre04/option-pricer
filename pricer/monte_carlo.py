"""
Pricer Monte Carlo pour options européennes sous le modèle de Black-Scholes
(mouvement brownien géométrique).
"""

from __future__ import annotations
import numpy as np


def _terminal_prices(S: float, T: float, r: float, sigma: float,
                      n_sims: int, q: float = 0.0, antithetic: bool = True,
                      seed: int | None = None) -> np.ndarray:
    """Simule les prix terminaux S_T sous la mesure risque-neutre."""
    rng = np.random.default_rng(seed)
    half = n_sims // 2 if antithetic else n_sims
    z = rng.standard_normal(half)

    if antithetic:
        z = np.concatenate([z, -z])

    drift = (r - q - 0.5 * sigma ** 2) * T
    diffusion = sigma * np.sqrt(T) * z
    return S * np.exp(drift + diffusion)


def mc_price(S: float, K: float, T: float, r: float, sigma: float,
             option_type: str = "call", q: float = 0.0,
             n_sims: int = 200_000, antithetic: bool = True,
             seed: int | None = None) -> dict:
    """
    Prix Monte Carlo + écart-type de l'estimateur (pour construire un IC à 95%).
    Renvoie un dict {"price": ..., "std_error": ..., "ci95": (low, high)}.
    """
    S_T = _terminal_prices(S, T, r, sigma, n_sims, q, antithetic, seed)

    if option_type == "call":
        payoffs = np.maximum(S_T - K, 0.0)
    elif option_type == "put":
        payoffs = np.maximum(K - S_T, 0.0)
    else:
        raise ValueError("option_type doit être 'call' ou 'put'.")

    discounted = np.exp(-r * T) * payoffs
    price = discounted.mean()
    std_error = discounted.std(ddof=1) / np.sqrt(len(discounted))

    return {
        "price": float(price),
        "std_error": float(std_error),
        "ci95": (float(price - 1.96 * std_error), float(price + 1.96 * std_error)),
    }


def mc_delta(S: float, K: float, T: float, r: float, sigma: float,
             option_type: str = "call", q: float = 0.0,
             n_sims: int = 200_000, bump: float = 0.01, seed: int = 42) -> float:
    """
    Delta Monte Carlo par différences finies centrées, avec germe aléatoire
    commun entre S+h et S-h pour que seul le bump change (réduction du bruit).
    """
    h = S * bump
    price_up = mc_price(S + h, K, T, r, sigma, option_type, q, n_sims, seed=seed)["price"]
    price_down = mc_price(S - h, K, T, r, sigma, option_type, q, n_sims, seed=seed)["price"]
    return (price_up - price_down) / (2 * h)


if __name__ == "__main__":
    try:
        from black_scholes import bs_price
    except ImportError:
        from pricer.black_scholes import bs_price

    S, K, T, r, sigma = 100, 100, 1.0, 0.03, 0.20
    bs = bs_price(S, K, T, r, sigma, "call")
    result = mc_price(S, K, T, r, sigma, "call", n_sims=500_000, seed=1)

    print(f"Black-Scholes      : {bs:.4f}")
    print(f"Monte Carlo (500k) : {result['price']:.4f}  "
          f"(IC95%: [{result['ci95'][0]:.4f}, {result['ci95'][1]:.4f}])")
    print(f"Delta MC vs analytique : {mc_delta(S, K, T, r, sigma, 'call'):.4f}")
