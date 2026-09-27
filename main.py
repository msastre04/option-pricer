"""
Démo complète du projet : compare Black-Scholes, arbre binomial et Monte Carlo,
affiche les Grecques, puis trace la smile de vol réelle.
"""

from pricer.black_scholes import bs_price, greeks
from pricer.binomial import crr_price
from pricer.monte_carlo import mc_price
from pricer.implied_vol import implied_vol


def section(title: str):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main():
    # Paramètres d'un call ATM à 1 an, exemple générique
    S, K, T, r, sigma = 100, 100, 1.0, 0.03, 0.20
    option_type = "call"

    section("1. Comparaison des 3 méthodes de pricing")
    price_bs = bs_price(S, K, T, r, sigma, option_type)
    price_crr = crr_price(S, K, T, r, sigma, N=500, option_type=option_type)
    result_mc = mc_price(S, K, T, r, sigma, option_type, n_sims=300_000, seed=1)

    print(f"Black-Scholes (analytique) : {price_bs:.4f}")
    print(f"Binomial CRR (N=500)       : {price_crr:.4f}  (écart: {abs(price_crr - price_bs):.5f})")
    print(f"Monte Carlo (300k tirages) : {result_mc['price']:.4f}  "
          f"(IC95%: [{result_mc['ci95'][0]:.4f}, {result_mc['ci95'][1]:.4f}])")

    section("2. Grecques (Black-Scholes)")
    g = greeks(S, K, T, r, sigma, option_type)
    for name, value in g.items():
        print(f"  {name:6s} : {value:+.5f}")

    section("3. Vérification de la volatilité implicite")
    recovered = implied_vol(price_bs, S, K, T, r, option_type)
    print(f"Vol injectée : {sigma:.4f}  |  Vol retrouvée par inversion : {recovered:.4f}")

    section("4. Smile de volatilité sur données réelles (SPY)")
    try:
        from pricer.smile import fetch_chain, plot_smile
        df = fetch_chain("SPY")
        print(df[["strike", "mid_price", "implied_vol", "moneyness"]].head(10).to_string(index=False))
        plot_smile(df, "SPY", save_path="smile_SPY.png")
    except Exception as e:
        print(f"(Smile non générée — pas de connexion ou données indisponibles : {e})")

    section("5. Surface de volatilité multi-échéances (SPY)")
    try:
        from pricer.vol_surface import fetch_surface, plot_smile_multi, plot_surface_3d
        df_surf = fetch_surface("SPY", max_expiries=6)
        print(df_surf.groupby("expiration")["implied_vol"].mean())
        plot_smile_multi(df_surf, "SPY", save_path="smiles_multi_SPY.png")
        plot_surface_3d(df_surf, "SPY", save_path="surface_SPY.png")
    except Exception as e:
        print(f"(Surface non générée — pas de connexion ou données indisponibles : {e})")

    print("\nTerminé.")


if __name__ == "__main__":
    main()
