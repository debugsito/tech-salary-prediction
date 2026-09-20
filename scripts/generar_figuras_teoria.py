#!/usr/bin/env python3
"""Genera las figuras didácticas del capítulo II (bases teóricas).

A diferencia de `generar_figuras.py`, estas figuras no proceden de los
artefactos de resultados: son ilustraciones esquemáticas de los conceptos
(sesgo-varianza, boosting, suavizado de la codificación, validación cruzada)
construidas con datos sintéticos de semilla fija. Cada pie de figura del
documento lo declara.

Uso:
    python scripts/generar_figuras_teoria.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

AZUL, NARANJA, VERDE, AMBAR = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
REJILLA = "#e5e4df"

ANCHO = 7.0
FIGDIR = Path(__file__).resolve().parent.parent / "figuras"

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.edgecolor": TINTA_2,
        "axes.labelcolor": TINTA,
        "xtick.color": TINTA_2,
        "ytick.color": TINTA_2,
        "axes.grid": True,
        "grid.color": REJILLA,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def fig_sesgo_varianza() -> None:
    """Curvas esquemáticas de sesgo, varianza y error total vs complejidad."""
    x = np.linspace(0.02, 1, 300)
    sesgo2 = 0.9 * np.exp(-4.2 * x)
    varianza = 0.04 * np.exp(3.1 * x)
    irreducible = np.full_like(x, 0.22)
    total = sesgo2 + varianza + irreducible

    fig, ax = plt.subplots(figsize=(ANCHO, 3.4))
    ax.plot(x, sesgo2, color=AZUL, lw=1.8, label="sesgo$^2$")
    ax.plot(x, varianza, color=NARANJA, lw=1.8, label="varianza")
    ax.plot(x, irreducible, color=TINTA_2, lw=1.4, ls="--", label="error irreducible $\\sigma^2$")
    ax.plot(x, total, color=TINTA, lw=2.4, label="error de generalización")
    xmin = x[np.argmin(total)]
    ax.axvline(xmin, color=VERDE, lw=1.2, ls=":")
    ax.annotate(
        "complejidad adecuada",
        xy=(xmin, total.min()),
        xytext=(xmin + 0.08, total.min() + 0.28),
        color=VERDE,
        fontsize=8.5,
        arrowprops=dict(arrowstyle="->", color=VERDE, lw=1),
    )
    ax.set_xlabel("Complejidad del modelo")
    ax.set_ylabel("Error esperado")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(frameon=False, loc="upper center", ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGDIR / "teoria_sesgo_varianza.png", dpi=200)
    plt.close(fig)


def fig_boosting() -> None:
    """Ajuste secuencial de un boosting de árboles sobre una señal sintética."""
    from sklearn.ensemble import GradientBoostingRegressor

    rng = np.random.default_rng(42)
    x = np.sort(rng.uniform(0, 10, 220))
    señal = np.sin(x) * 2.2 + 0.35 * x
    y = señal + rng.normal(0, 0.55, x.size)
    X = x.reshape(-1, 1)
    malla = np.linspace(0, 10, 400).reshape(-1, 1)

    fig, axes = plt.subplots(1, 3, figsize=(ANCHO, 2.7), sharey=True)
    for ax, m in zip(axes, (1, 10, 100)):
        gb = GradientBoostingRegressor(
            n_estimators=m, learning_rate=0.3, max_depth=2, random_state=42
        ).fit(X, y)
        ax.scatter(x, y, s=6, color=TINTA_2, alpha=0.45, linewidths=0)
        ax.plot(malla, np.sin(malla) * 2.2 + 0.35 * malla, color=VERDE, lw=1.2, ls="--")
        ax.plot(malla, gb.predict(malla), color=NARANJA, lw=1.9)
        ax.set_title(f"$M = {m}$ árboles", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].set_ylabel("respuesta")
    fig.tight_layout()
    fig.savefig(FIGDIR / "teoria_boosting.png", dpi=200)
    plt.close(fig)


def fig_encoding() -> None:
    """Peso del suavizado lambda(n) = n / (n + m) para varios m."""
    n = np.linspace(1, 300, 400)
    fig, ax = plt.subplots(figsize=(ANCHO, 3.0))
    for m, color in ((5, AZUL), (20, NARANJA), (50, VERDE), (100, AMBAR)):
        ax.plot(n, n / (n + m), lw=1.8, color=color, label=f"$m = {m}$")
    ax.axhline(1.0, color=REJILLA, lw=0.8)
    ax.set_xlabel("Observaciones del nivel categórico ($n_c$)")
    ax.set_ylabel("Peso de la media propia $\\lambda(n_c)$")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, title="suavizado", fontsize=8, title_fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGDIR / "teoria_codificacion.png", dpi=200)
    plt.close(fig)


def fig_cv() -> None:
    """Esquema de validación cruzada de 5 particiones con repetición."""
    reps, k = 4, 5
    fig, axes = plt.subplots(reps, 1, figsize=(ANCHO, 3.2))
    for r, ax in enumerate(axes):
        for i in range(k):
            for j in range(k):
                color = NARANJA if i == j else "#dce8f7"
                ax.barh(k - 1 - i, 1, left=j, height=0.82, color=color, edgecolor="white", lw=1.2)
        ax.set_xlim(0, k + 2.2)
        ax.set_ylim(-0.6, k - 0.4)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for lado in ("left", "bottom"):
            ax.spines[lado].set_visible(False)
        ax.text(k + 0.25, (k - 1) / 2, f"repetición {r + 1}", va="center", fontsize=8.5, color=TINTA_2)
    axes[0].text(0, k + 0.4, "bloques de la muestra →", fontsize=8.5, color=TINTA_2)
    axes[-1].barh(-2, 0, color=NARANJA, label="validación")
    axes[-1].barh(-2, 0, color="#dce8f7", label="entrenamiento")
    axes[-1].legend(frameon=False, loc="lower right", ncol=2, fontsize=8, bbox_to_anchor=(1.0, -0.75))
    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(FIGDIR / "teoria_cv.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    FIGDIR.mkdir(exist_ok=True)
    fig_sesgo_varianza()
    fig_boosting()
    fig_encoding()
    fig_cv()
    print("figuras del capítulo II generadas en", FIGDIR)
