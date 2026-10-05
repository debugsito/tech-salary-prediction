#!/usr/bin/env python3
"""Contribución SHAP agregada por bloque teórico, con intervalos por remuestreo.

Usa los valores SHAP ya calculados por ejecutar_analisis.py y la misma
asignación de bloques que ejecutar_ppp.py y ejecutar_consistencia.py, de modo
que las tres cifras sean comparables. Escribe resultados/shap_bloques_<año>.json.

Uso:
    python scripts/ejecutar_bloques_shap.py              # 2022, 2023 y 2025
    python scripts/ejecutar_bloques_shap.py --anio 2023
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
RES = RAIZ / "resultados"
SEMILLA = 42
REPLICAS = 1000


def bloque(v: str) -> str:
    if v.startswith(("language__", "database__", "platform__", "webframe__")):
        return "Tecnologías"
    if v.startswith(("Country", "income_group", "wb_region")):
        return "Geográfico"
    if v.startswith(("YearsCode", "WorkExp", "EdLevel", "DevType", "ICorPM")):
        return "Capital humano"
    if v.startswith(("Age", "Gender")):
        return "Demográfico"
    if v.startswith(("AISelect", "AIThreat", "AIAgents", "LearnCodeAI")):
        return "Adopción de IA"
    return "Organizacional"


def contribuciones(valores: np.ndarray, bloques: np.ndarray, nombres: list[str]) -> np.ndarray:
    medias = np.abs(valores).mean(axis=0)
    return np.array([medias[bloques == b].sum() for b in nombres])


def calcular(anio: str) -> dict:
    datos = np.load(RES / f"shap_values_XGBoost_target_{anio}.npz", allow_pickle=True)
    valores = datos["shap_values"]
    bloques = np.array([bloque(str(v)) for v in datos["feature_names"]])
    nombres = sorted(set(bloques))

    puntual = contribuciones(valores, bloques, nombres)
    rng = np.random.default_rng(SEMILLA)
    n = len(valores)
    remuestras = np.array([
        contribuciones(valores[rng.integers(0, n, n)], bloques, nombres)
        for _ in range(REPLICAS)])
    inf, sup = np.percentile(remuestras, [2.5, 97.5], axis=0)

    salida = {
        "anio": anio, "n_observaciones": int(n), "replicas": REPLICAS, "semilla": SEMILLA,
        "bloques": {
            b: {"contribucion": round(float(p), 4), "ic_inferior": round(float(i), 4),
                "ic_superior": round(float(s), 4),
                "n_variables": int((bloques == b).sum())}
            for b, p, i, s in sorted(zip(nombres, puntual, inf, sup), key=lambda t: -t[1])
        },
    }
    (RES / f"shap_bloques_{anio}.json").write_text(
        json.dumps(salida, ensure_ascii=False, indent=2), encoding="utf-8")
    return salida


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--anio", choices=["2022", "2023", "2025"])
    args = ap.parse_args()
    for anio in ([args.anio] if args.anio else ["2022", "2023", "2025"]):
        s = calcular(anio)
        print(anio)
        for b, v in s["bloques"].items():
            print(f"  {b:16s} {v['contribucion']:.4f}  [{v['ic_inferior']:.4f}, {v['ic_superior']:.4f}]"
                  f"  ({v['n_variables']} variables)")
