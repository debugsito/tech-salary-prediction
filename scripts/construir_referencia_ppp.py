#!/usr/bin/env python3
"""Construye la tabla de factores de paridad de poder adquisitivo por país.

El análisis principal opera en dólares nominales, y la sección 5.3.5 del
documento reconoce que esa escala no distingue la dispersión atribuible al
nivel de precios de la atribuible a segmentación del mercado. Esta tabla
permite reexpresar la compensación en dólares internacionales (PPA) y medir
esa distinción en lugar de declararla pendiente.

Se emplea el factor de conversión de **consumo privado** (PA.NUS.PRVT.PP) y no
el del PIB, porque lo que se compara es el poder de compra de un salario en
manos de un hogar, no la producción de una economía. El cociente entre ese
factor y el tipo de cambio oficial (PA.NUS.FCRF) da el nivel de precios
relativo a Estados Unidos:

    salario_ppa = salario_nominal_usd / (PRVT.PP / FCRF)

Para Estados Unidos ambos factores valen uno y el salario no cambia.

La tabla se versiona como artefacto, igual que la de niveles de renta: el Banco
Mundial revisa estas series y un resultado que dependiera de la fecha de
ejecución no sería reproducible.

Uso:
    python scripts/construir_referencia_ppp.py
"""

from __future__ import annotations

import csv
import json
import sys
import urllib.request
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "scripts"))

from construir_referencia_paises import ALIAS_ENCUESTA  # noqa: E402

SALIDA = RAIZ / "datos" / "referencia" / "factores_ppp.csv"

# Año de referencia: el de la edición principal del estudio. Si un país no
# tiene dato para ese año se acepta el más reciente dentro de la ventana, y la
# tabla registra qué año se usó para que la sustitución quede a la vista.
ANIO = 2023
VENTANA = ["2023", "2022", "2021"]

API = ("https://api.worldbank.org/v2/country/all/indicator/{ind}"
       "?format=json&date=2021:2023&per_page=20000")


def descargar_indicador(codigo: str) -> dict[str, dict[str, float]]:
    """Serie {pais_wb: {anio: valor}} de un indicador del Banco Mundial."""
    with urllib.request.urlopen(API.format(ind=codigo), timeout=60) as r:
        datos = json.load(r)[1]
    serie: dict[str, dict[str, float]] = {}
    for fila in datos:
        if fila["value"] is None:
            continue
        serie.setdefault(fila["country"]["value"], {})[fila["date"]] = float(fila["value"])
    return serie


def valor_reciente(serie: dict[str, float]) -> tuple[float, str] | None:
    for anio in VENTANA:
        if anio in serie:
            return serie[anio], anio
    return None


def paises_de_la_encuesta() -> list[str]:
    """Los países de la muestra efectiva, leídos del contexto ya exportado."""
    ctx = RAIZ / "servicio" / "artefactos" / "contexto.json"
    if ctx.exists():
        return sorted(json.loads(ctx.read_text(encoding="utf-8"))
                      ["referencia"]["income_group"].keys())
    from src.carga_encuesta import cargar_encuesta
    df, _ = cargar_encuesta(anio="2023")
    return sorted(df["Country"].unique())


def main() -> int:
    print("Descargando PA.NUS.PRVT.PP (PPA de consumo privado)...")
    ppp = descargar_indicador("PA.NUS.PRVT.PP")
    print("Descargando PA.NUS.FCRF (tipo de cambio oficial)...")
    fcrf = descargar_indicador("PA.NUS.FCRF")

    filas, sin_dato = [], []
    for pais in paises_de_la_encuesta():
        nombre_wb = ALIAS_ENCUESTA.get(pais, pais)
        p, f = ppp.get(nombre_wb), fcrf.get(nombre_wb)
        vp = valor_reciente(p) if p else None
        vf = valor_reciente(f) if f else None
        if not vp or not vf:
            sin_dato.append(pais)
            continue
        nivel_precios = vp[0] / vf[0]
        filas.append({
            "pais": pais,
            "ppp_consumo": round(vp[0], 6),
            "tipo_cambio": round(vf[0], 6),
            "nivel_precios": round(nivel_precios, 4),
            "anio_ppp": vp[1],
            "anio_tc": vf[1],
        })

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    with SALIDA.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)

    print(f"\n{SALIDA.relative_to(RAIZ)}: {len(filas)} países")
    if sin_dato:
        print(f"Sin factor PPA ({len(sin_dato)}): {', '.join(sin_dato)}")
    ejemplo = {f["pais"]: f["nivel_precios"] for f in filas
               if f["pais"] in ("United States of America", "Germany", "Brazil",
                                 "India", "Peru")}
    print("Nivel de precios relativo a EE. UU.:", ejemplo)
    return 0


if __name__ == "__main__":
    sys.exit(main())
