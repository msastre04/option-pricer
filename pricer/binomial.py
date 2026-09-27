"""
Arbre binomial de Cox-Ross-Rubinstein (CRR).

Sert à deux choses dans ce projet :
1. Vérifier la convergence vers Black-Scholes quand N (nombre de pas) augmente.
2. Pricer des options américaines (exercice anticipé possible), ce que
   Black-Scholes ne sait pas faire nativement.
"""

from __future__ import annotations
import numpy as np


def crr_price(S: float, K: float, T: float, r: float, sigma: float,
              N: int = 200, option_type: str = "call",
              american: bool = False, q: float = 0.0) -> float:
    """
    Prix d'une option via un arbre binomial CRR à N pas.

    american=True  -> possibilité d'exercice à chaque noeud (option américaine)
    american=False -> exercice uniquement à maturité (option européenne)
    """
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))          # facteur de hausse
    d = 1 / u                                 # facteur de baisse
    disc = np.exp(-r * dt)
    p = (np.exp((r - q) * dt) - d) / (u - d)  # probabilité risque-neutre

    if not (0 < p < 1):
        raise ValueError("Probabilité risque-neutre hors [0,1] : vérifier les paramètres (dt trop grand ?).")

    # Prix du sous-jacent à maturité pour chaque noeud final (j = nb de hausses)
    j = np.arange(N + 1)
    S_T = S * (u ** j) * (d ** (N - j))

    if option_type == "call":
        values = np.maximum(S_T - K, 0.0)
    elif option_type == "put":
        values = np.maximum(K - S_T, 0.0)
    else:
        raise ValueError("option_type doit être 'call' ou 'put'.")

    # Remontée dans l'arbre (backward induction)
    for step in range(N - 1, -1, -1):
        values = disc * (p * values[1:step + 2] + (1 - p) * values[0:step + 1])

        if american:
            j_step = np.arange(step + 1)
            S_t = S * (u ** j_step) * (d ** (step - j_step))
            if option_type == "call":
                exercise = np.maximum(S_t - K, 0.0)
            else:
                exercise = np.maximum(K - S_t, 0.0)
            values = np.maximum(values, exercise)

    return float(values[0])


if __name__ == "__main__":
    try:
        from black_scholes import bs_price
    except ImportError:
        from pricer.black_scholes import bs_price

    S, K, T, r, sigma = 100, 100, 1.0, 0.03, 0.20
    bs = bs_price(S, K, T, r, sigma, "call")

    print("Convergence de l'arbre CRR vers Black-Scholes (call européen) :")
    for N in [10, 50, 200, 1000]:
        crr = crr_price(S, K, T, r, sigma, N=N, option_type="call", american=False)
        print(f"  N={N:5d}  CRR={crr:.4f}  BS={bs:.4f}  écart={abs(crr - bs):.5f}")

    put_am = crr_price(S, K, T, r, sigma, N=500, option_type="put", american=True)
    put_eu = crr_price(S, K, T, r, sigma, N=500, option_type="put", american=False)
    print(f"\nPut américain : {put_am:.4f}  |  Put européen : {put_eu:.4f}  "
          f"(prime d'exercice anticipé : {put_am - put_eu:.4f})")
