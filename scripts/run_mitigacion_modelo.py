#!/usr/bin/env python3
"""Mitigación a nivel de modelo: calibración por grupo y pérdida igualada.

La sección 5.4.6 descartó las estrategias que actúan sobre la composición de la
muestra. Quedaban por evaluar las que actúan sobre el modelo, que el trabajo
había dejado como línea futura:

1. **Calibración posterior por grupo.** Se corrige la predicción con el sesgo
   (aditiva) o con una recta (afín) ajustados por nivel de renta sobre el
   conjunto de entrenamiento, sin tocar el modelo.
2. **Reponderación iterativa hacia pérdida igualada.** Aproximación al esquema
   de reducciones de Agarwal et al. (2019): en cada iteración se reentrena con
   pesos por grupo y se aumenta el peso de los grupos con mayor error relativo,
   buscando igualar la pérdida entre grupos. Se reporta la trayectoria completa
   de compromiso entre exactitud y disparidad, no solo el punto final.

Todo se evalúa sobre el mismo conjunto de prueba y con los mismos indicadores
que las estrategias de composición, para que ambas familias sean comparables.

Artefactos: results/mitigacion_modelo.json

Uso:
    python scripts/run_mitigacion_modelo.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.data_loader_stackoverflow import cargar_encuesta  # noqa: E402
from src.fairness import auditar  # noqa: E402
from src.feature_pipeline_so import construir_preprocesador, preparar_xy  # noqa: E402
from src.metrics import compute_metrics  # noqa: E402
from src.model_registry import build_models  # noqa: E402

SEMILLA = 42
SALIDA = RAIZ / "results" / "mitigacion_modelo.json"
ITERACIONES = 8
ETA = 0.5  # tasa de actualización de los pesos por grupo


def evaluar(nombre, y_pru, pred, grupos_pru):
    m = compute_metrics(y_pru, pred)
    aud = auditar(y_pru, pred, grupos_pru)
    return {
        "estrategia": nombre,
        "R2": round(float(m["R2"]), 4),
        "MAE_USD": round(float(m["MAE_USD"])),
        "razon_disparidad_renta": round(
            aud["agregados"]["razon_disparidad_relativa"], 3),
        "error_relativo": {g: round(v["mae_relativo"], 4)
                          for g, v in aud["por_grupo"].items()},
    }


def main() -> int:
    df, _ = cargar_encuesta(anio="2023")
    X, y, A = preparar_xy(df)
    y = df["salary_log"].to_numpy()
    estratos = df["income_group"].astype(str)
    X_ent, X_pru, y_ent, y_pru, A_ent, A_pru = train_test_split(
        X, y, A, test_size=0.20, random_state=SEMILLA, stratify=estratos)
    g_ent = A_ent["income_group"].to_numpy()
    g_pru = A_pru["income_group"]

    def entrenar(pesos=None):
        t = Pipeline([("preprocesador", construir_preprocesador(df, "target")),
                      ("modelo", build_models()["XGBoost"])])
        if pesos is None:
            t.fit(X_ent, y_ent)
        else:
            t.fit(X_ent, y_ent, modelo__sample_weight=pesos)
        return t

    base = entrenar()
    pred_ent = base.predict(X_ent)
    pred_pru = base.predict(X_pru)
    filas = [evaluar("sin_mitigacion", y_pru, pred_pru, g_pru)]
    print(f"base: R²={filas[0]['R2']}  razón={filas[0]['razon_disparidad_renta']}")

    # 1a. Calibración aditiva por grupo: el sesgo logarítmico del entrenamiento.
    correccion = {g: float(np.mean(y_ent[g_ent == g] - pred_ent[g_ent == g]))
                  for g in np.unique(g_ent)}
    pred_cal = pred_pru + g_pru.map(correccion).to_numpy()
    filas.append(evaluar("calibracion_aditiva", y_pru, pred_cal, g_pru))

    # 1b. Calibración afín por grupo: recta ajustada sobre el entrenamiento.
    pred_afin = np.empty_like(pred_pru)
    coefs = {}
    for g in np.unique(g_ent):
        sel = g_ent == g
        b, a = np.polyfit(pred_ent[sel], y_ent[sel], 1)
        coefs[g] = (round(float(a), 4), round(float(b), 4))
        sel_p = (g_pru == g).to_numpy()
        pred_afin[sel_p] = a + b * pred_pru[sel_p]
    filas.append(evaluar("calibracion_afin", y_pru, pred_afin, g_pru))

    # 2. Reponderación iterativa hacia pérdida igualada (Agarwal et al., 2019,
    # en versión simplificada). El error relativo por grupo se mide sobre el
    # propio entrenamiento para no tocar la prueba durante la búsqueda.
    grupos = list(np.unique(g_ent))
    w_g = {g: 1.0 for g in grupos}
    trayectoria = []
    for i in range(ITERACIONES):
        pesos = pd.Series(g_ent).map(w_g).to_numpy()
        pesos = pesos / pesos.mean()
        t = entrenar(pesos)
        pred_e = t.predict(X_ent)
        # Error relativo por grupo en escala original, sobre entrenamiento.
        reales, predichos = np.expm1(y_ent), np.expm1(pred_e)
        err = {}
        for g in grupos:
            sel = g_ent == g
            err[g] = float(np.abs(predichos[sel] - reales[sel]).mean()
                           / np.median(reales[sel]))
        media = float(np.mean(list(err.values())))
        for g in grupos:
            w_g[g] *= float(np.exp(ETA * (err[g] - media) / media))
        punto = evaluar(f"perdida_igualada_iter_{i + 1}", y_pru,
                        t.predict(X_pru), g_pru)
        punto["pesos"] = {g: round(w_g[g], 3) for g in grupos}
        trayectoria.append(punto)
        print(f"  iter {i + 1}: R²={punto['R2']}  "
              f"razón={punto['razon_disparidad_renta']}  pesos={punto['pesos']}")

    mejor_iter = min(trayectoria, key=lambda p: p["razon_disparidad_renta"])
    filas.append({**mejor_iter, "estrategia": "perdida_igualada_mejor"})

    SALIDA.write_text(json.dumps({
        "correccion_aditiva_log": {k: round(v, 4) for k, v in correccion.items()},
        "coeficientes_afines": coefs,
        "iteraciones": ITERACIONES, "eta": ETA,
        "estrategias": filas,
        "trayectoria": trayectoria,
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n{SALIDA.relative_to(RAIZ)} escrito\n")
    print(f"{'estrategia':26} {'R²':>7} {'MAE':>8} {'razón':>6}  error relativo por grupo")
    for f in filas:
        print(f"{f['estrategia']:26} {f['R2']:>7} {f['MAE_USD']:>8,} "
              f"{f['razon_disparidad_renta']:>6}  "
              + "  ".join(f"{g[:12]}:{v}" for g, v in f["error_relativo"].items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
