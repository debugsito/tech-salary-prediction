#!/usr/bin/env python3
"""Genera el diagrama de arquitectura del prototipo (sección 4.5.4).

Dos carriles: el flujo de una consulta en operación y el flujo de
construcción de la imagen, donde vive la comprobación de paridad.

Uso:
    python scripts/generar_figura_arquitectura.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

AZUL, NARANJA, VERDE = "#2a78d6", "#eb6834", "#1baf7a"
TINTA, TINTA_2, PAPEL = "#0b0b0b", "#52514e", "#f6f5f0"

FIGDIR = Path(__file__).resolve().parent.parent / "figuras"


def caja(ax, x, y, w, h, titulo, lineas=(), borde=TINTA_2, fondo="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                fc=fondo, ec=borde, lw=1.3))
    ax.text(x + w / 2, y + h - 0.14, titulo, ha="center", va="top",
            fontsize=9, fontweight="bold", color=TINTA)
    for i, l in enumerate(lineas):
        ax.text(x + w / 2, y + h - 0.46 - i * 0.26, l, ha="center", va="top",
                fontsize=7.6, color=TINTA_2, family="monospace")


def flecha(ax, x0, y0, x1, y1, texto="", color=TINTA_2, dx=0.0, dy=0.09):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=13, color=color, lw=1.4))
    if texto:
        ax.text((x0 + x1) / 2 + dx, (y0 + y1) / 2 + dy, texto, ha="center",
                fontsize=7.4, color=color, family="monospace")


def main() -> None:
    fig, ax = plt.subplots(figsize=(7.0, 3.9))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.6)
    ax.axis("off")

    # Carril superior: operación.
    ax.text(0.15, 5.38, "Operación", fontsize=9.5, color=TINTA, fontweight="bold")
    caja(ax, 0.15, 3.55, 2.1, 1.5, "Navegador", ["página estática", "sin dependencias"])
    caja(ax, 3.35, 3.55, 2.2, 1.5, "Caddy", ["proxy inverso", "certificados TLS"])
    caja(ax, 6.55, 3.55, 3.3, 1.5, "Contenedor",
         ["FastAPI + compuerta", "modelo.joblib 0.5 MB", "contexto.json"], borde=VERDE)
    flecha(ax, 2.25, 4.3, 3.35, 4.3, "HTTPS")
    flecha(ax, 5.55, 4.3, 6.55, 4.3, "red int.")

    # Carril inferior: construcción.
    ax.text(0.15, 2.62, "Construcción de la imagen", fontsize=9.5, color=TINTA, fontweight="bold")
    caja(ax, 0.15, 0.95, 2.1, 1.5, "Repositorio", ["servicio/ + src/", "artefactos"])
    caja(ax, 3.05, 0.95, 3.0, 1.5, "docker build",
         ["dependencias fijadas", "24 pruebas dentro", "de la imagen"],
         borde=NARANJA, fondo=PAPEL)
    caja(ax, 7.05, 0.95, 2.3, 1.5, "Imagen", ["python 3.12-slim", "usuario sin", "privilegios"])
    flecha(ax, 2.25, 1.7, 3.05, 1.7)
    flecha(ax, 6.05, 1.7, 7.05, 1.7, "si pasan", color=NARANJA, dy=0.14)

    # La imagen publicada alimenta el contenedor en operación.
    flecha(ax, 8.2, 2.45, 8.2, 3.55, "despliegue", dx=0.85, dy=0.0)

    ax.text(5.0, 0.35,
            "La comprobación de paridad corre al construir: una imagen que no\n"
            "reproduzca las cifras del entrenamiento no llega a existir.",
            ha="center", fontsize=7.8, color=TINTA_2, style="italic")

    fig.tight_layout()
    fig.savefig(FIGDIR / "arquitectura_servicio.png", dpi=200)
    print("arquitectura_servicio.png generada")


if __name__ == "__main__":
    main()
