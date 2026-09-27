"""
Surface de vol implicite multi-échéances : on récupère plusieurs expirations
sur la chaîne d'options, on calcule l'IV par strike pour chacune, et on sort
soit les smiles superposées (2D), soit une surface interpolée (3D).
"""

from __future__ import annotations
from datetime import date
import numpy as np
import pandas as pd

try:
    from implied_vol import implied_vol
except ImportError:
    from pricer.implied_vol import implied_vol


def fetch_surface(ticker: str, r: float = 0.04, option_type: str = "call",
                   max_expiries: int = 6, min_days: int = 7, max_days: int = 365) -> pd.DataFrame:
    """Récupère plusieurs échéances et calcule la vol implicite par strike pour chacune."""
    import yfinance as yf

    tk = yf.Ticker(ticker)
    spot = tk.history(period="1d")["Close"].iloc[-1]
    today = date.today()

    expirations = [e for e in tk.options
                   if min_days <= (date.fromisoformat(e) - today).days <= max_days]
    expirations = expirations[:max_expiries]

    if not expirations:
        raise RuntimeError(f"Aucune échéance dans la fenêtre {min_days}-{max_days} jours pour {ticker}.")

    rows = []
    for exp in expirations:
        T = (date.fromisoformat(exp) - today).days / 365
        chain = tk.option_chain(exp)
        df = chain.calls if option_type == "call" else chain.puts
        df = df.copy()
        df["mid_price"] = (df["bid"] + df["ask"]) / 2
        df = df[df["mid_price"] > 0]

        for _, row in df.iterrows():
            iv = implied_vol(row["mid_price"], spot, row["strike"], T, r, option_type)
            if iv is not None:
                rows.append({
                    "expiration": exp,
                    "T": T,
                    "strike": row["strike"],
                    "moneyness": row["strike"] / spot,
                    "implied_vol": iv,
                })

    return pd.DataFrame(rows)


def plot_smile_multi(df: pd.DataFrame, ticker: str, save_path: str | None = None):
    """Une courbe de smile par échéance, superposées, couleur = maturité."""
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    fig, ax = plt.subplots(figsize=(9, 6))
    expirations = sorted(df["expiration"].unique(), key=lambda e: date.fromisoformat(e))
    colors = cm.viridis(np.linspace(0, 1, len(expirations)))

    for exp, color in zip(expirations, colors):
        sub = df[df["expiration"] == exp].sort_values("moneyness")
        days = int(sub["T"].iloc[0] * 365)
        ax.plot(sub["moneyness"], sub["implied_vol"] * 100, "o-", color=color,
                markersize=3, label=f"{exp} ({days}j)")

    ax.axvline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Moneyness (K / Spot)")
    ax.set_ylabel("Volatilité implicite (%)")
    ax.set_title(f"Smiles par échéance — {ticker}")
    ax.legend(title="Échéance", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Graphique sauvegardé : {save_path}")
    else:
        plt.show()
    return fig


def plot_surface_3d(df: pd.DataFrame, ticker: str, save_path: str | None = None):
    """Surface 3D interpolée (moneyness x maturité x vol implicite)."""
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata

    x = df["moneyness"].values
    y = df["T"].values * 365
    z = df["implied_vol"].values * 100

    grid_x, grid_y = np.meshgrid(
        np.linspace(x.min(), x.max(), 60),
        np.linspace(y.min(), y.max(), 60),
    )
    grid_z = griddata((x, y), z, (grid_x, grid_y), method="linear")

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap="viridis", edgecolor="none", alpha=0.9)
    ax.scatter(x, y, z, color="black", s=8, alpha=0.5)

    ax.set_xlabel("Moneyness (K / Spot)")
    ax.set_ylabel("Maturité (jours)")
    ax.set_zlabel("Vol implicite (%)")
    ax.set_title(f"Surface de volatilité — {ticker}")
    fig.colorbar(surf, shrink=0.6, label="Vol implicite (%)")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Graphique sauvegardé : {save_path}")
    else:
        plt.show()
    return fig


if __name__ == "__main__":
    ticker = "SPY"
    df = fetch_surface(ticker)
    print(df.groupby("expiration")["implied_vol"].describe())
    plot_smile_multi(df, ticker, save_path="smiles_multi_" + ticker + ".png")
    plot_surface_3d(df, ticker, save_path="surface_" + ticker + ".png")
