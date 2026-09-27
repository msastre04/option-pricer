"""
Récupère une vraie chaîne d'options via yfinance et calcule la volatilité
implicite pour chaque strike -> "smile" (ou "skew") de volatilité.
"""

from __future__ import annotations
from datetime import date
import numpy as np
import pandas as pd

try:
    from implied_vol import implied_vol
except ImportError:
    from pricer.implied_vol import implied_vol


def fetch_chain(ticker: str, r: float = 0.04, option_type: str = "call") -> pd.DataFrame:
    """
    Va chercher la chaîne d'options la plus proche de 30-60 jours pour `ticker`
    et calcule la vol implicite de chaque strike à partir du prix mid marché.

    Renvoie un DataFrame avec colonnes : strike, mid_price, T, implied_vol, moneyness.
    """
    import yfinance as yf

    tk = yf.Ticker(ticker)
    spot = tk.history(period="1d")["Close"].iloc[-1]

    expirations = tk.options
    if not expirations:
        raise RuntimeError(f"Pas de chaîne d'options disponible pour {ticker}.")

    today = date.today()
    # on choisit l'échéance la plus proche de 45 jours (liquidité correcte, peu de bruit)
    target_days = 45
    best_exp = min(expirations, key=lambda e: abs(
        (date.fromisoformat(e) - today).days - target_days))
    T = max((date.fromisoformat(best_exp) - today).days, 1) / 365

    chain = tk.option_chain(best_exp)
    df = chain.calls if option_type == "call" else chain.puts

    df = df.copy()
    df["mid_price"] = (df["bid"] + df["ask"]) / 2
    df = df[(df["mid_price"] > 0) & (df["volume"].fillna(0) >= 0)]

    ivs = []
    for _, row in df.iterrows():
        iv = implied_vol(row["mid_price"], spot, row["strike"], T, r, option_type)
        ivs.append(iv)
    df["implied_vol"] = ivs
    df["T"] = T
    df["spot"] = spot
    df["moneyness"] = df["strike"] / spot
    df["expiration"] = best_exp

    return df.dropna(subset=["implied_vol"]).sort_values("strike").reset_index(drop=True)


def plot_smile(df: pd.DataFrame, ticker: str, save_path: str | None = None):
    """Trace la smile (vol implicite en fonction du strike / moneyness)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["moneyness"], df["implied_vol"] * 100, "o-", color="#1f4e79")
    ax.axvline(1.0, color="gray", linestyle="--", linewidth=1, label="ATM (moneyness=1)")
    ax.set_xlabel("Moneyness (K / Spot)")
    ax.set_ylabel("Volatilité implicite (%)")
    ax.set_title(f"Smile de volatilité — {ticker} (échéance {df['expiration'].iloc[0]})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Graphique sauvegardé : {save_path}")
    else:
        plt.show()

    return fig


if __name__ == "__main__":
    ticker = "SPY"
    df = fetch_chain(ticker)
    print(df[["strike", "mid_price", "implied_vol", "moneyness"]].to_string(index=False))
    plot_smile(df, ticker, save_path="smile_" + ticker + ".png")
