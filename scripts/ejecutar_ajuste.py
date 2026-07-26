#!/usr/bin/env python3
"""Optimización de hiperparámetros con validación anidada.

La búsqueda se realiza **exclusivamente sobre el conjunto de entrenamiento**,
mediante una validación cruzada interna. El conjunto de prueba no interviene en
ninguna decisión de configuración: si la selección de hiperparámetros empleara
esas observaciones, la evaluación posterior sobre ellas dejaría de ser
independiente y las métricas resultantes estarían sesgadas al alza.

Este procedimiento subsana la limitación declarada en §6.2.2, donde se hacía
constar que los modelos se habían evaluado con configuraciones por defecto.

Salida:
    resultados/ajuste_resultados.csv   Configuración óptima y desempeño por modelo
    resultados/ajuste_espacio.csv      Todas las combinaciones evaluadas

Uso:
    python scripts/ejecutar_ajuste.py                    # búsqueda completa
    python scripts/ejecutar_ajuste.py --iteraciones 20   # búsqueda más breve
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.linear_model import Ridge
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     train_test_split)
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.carga_encuesta import cargar_encuesta  # noqa: E402
from src.tuberia_caracteristicas import construir_preprocesador, preparar_xy  # noqa: E402

SEMILLA = 42
PROPORCION_PRUEBA = 0.20
N_PLIEGUES_INTERNOS = 3

# Espacios de búsqueda. Se emplea muestreo aleatorio y no rejilla exhaustiva:
# con el mismo presupuesto de cómputo, el muestreo aleatorio explora mejor los
# espacios en los que solo unos pocos hiperparámetros son determinantes.
ESPACIOS = {
    "Ridge": {
        "modelo__alpha": loguniform(1e-3, 1e3),
    },
    "XGBoost": {
        "modelo__n_estimators": randint(200, 1200),
        "modelo__max_depth": randint(3, 12),
        "modelo__learning_rate": loguniform(0.01, 0.3),
        "modelo__subsample": uniform(0.6, 0.4),
        "modelo__colsample_bytree": uniform(0.5, 0.5),
        "modelo__min_child_weight": randint(1, 20),
        "modelo__reg_lambda": loguniform(0.1, 20),
        "modelo__reg_alpha": loguniform(1e-3, 5),
    },
    "LightGBM": {
        "modelo__n_estimators": randint(200, 1200),
        "modelo__num_leaves": randint(16, 256),
        "modelo__learning_rate": loguniform(0.01, 0.3),
        "modelo__subsample": uniform(0.6, 0.4),
        "modelo__colsample_bytree": uniform(0.5, 0.5),
        "modelo__min_child_samples": randint(5, 100),
        "modelo__reg_lambda": loguniform(0.1, 20),
    },
    "HistGradientBoosting": {
        "modelo__max_iter": randint(200, 900),
        "modelo__max_depth": randint(3, 15),
        "modelo__learning_rate": loguniform(0.01, 0.3),
        "modelo__max_leaf_nodes": randint(15, 128),
        "modelo__min_samples_leaf": randint(5, 100),
        "modelo__l2_regularization": loguniform(1e-4, 10),
    },
}


def construir_base(nombre: str):
    """Instancia sin configurar del modelo, para que la búsqueda la parametrice."""
    if nombre == "Ridge":
        return Ridge()
    if nombre == "XGBoost":
        from xgboost import XGBRegressor
        return XGBRegressor(random_state=SEMILLA, n_jobs=-1,
                            objective="reg:squarederror", enable_categorical=True)
    if nombre == "LightGBM":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(random_state=SEMILLA, n_jobs=-1, verbose=-1)
    if nombre == "HistGradientBoosting":
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(random_state=SEMILLA)
    raise ValueError(nombre)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--anio", default="2023")
    ap.add_argument("--codificacion", default="target")
    ap.add_argument("--modelos", nargs="+", default=list(ESPACIOS))
    ap.add_argument("--iteraciones", type=int, default=40)
    ap.add_argument("--out-dir", default="resultados")
    args = ap.parse_args()

    t0 = time.perf_counter()
    df, _ = cargar_encuesta(anio=args.anio)
    X, y, _ = preparar_xy(df)

    X_ent, X_pru, y_ent, y_pru, e_ent, _ = train_test_split(
        X, y, df["income_group"], test_size=PROPORCION_PRUEBA,
        random_state=SEMILLA, stratify=df["income_group"])
    print(f"Muestra: {len(df):,} · entrenamiento {len(X_ent):,} · prueba {len(X_pru):,}")
    print(f"Búsqueda: {args.iteraciones} combinaciones × {N_PLIEGUES_INTERNOS} pliegues internos\n")

    interna = StratifiedKFold(n_splits=N_PLIEGUES_INTERNOS, shuffle=True,
                              random_state=SEMILLA)

    filas, espacio = [], []
    for nombre in args.modelos:
        if nombre not in ESPACIOS:
            print(f"[{nombre}] sin espacio de búsqueda definido; omitido")
            continue
        t = time.perf_counter()
        print(f"[{nombre}] buscando...")

        tuberia = Pipeline([
            ("preprocesador", construir_preprocesador(df, args.codificacion)),
            ("modelo", construir_base(nombre)),
        ])
        busqueda = RandomizedSearchCV(
            tuberia, ESPACIOS[nombre], n_iter=args.iteraciones,
            scoring="r2", cv=interna.split(X_ent, e_ent),
            random_state=SEMILLA, n_jobs=1, refit=True, error_score="raise",
        )
        busqueda.fit(X_ent, y_ent)

        r2_prueba = busqueda.best_estimator_.score(X_pru, y_pru)
        pred = busqueda.best_estimator_.predict(X_pru)
        mae_usd = float(np.abs(np.expm1(y_pru) - np.expm1(pred)).mean())

        limpio = {k.replace("modelo__", ""): (round(v, 5) if isinstance(v, float) else v)
                  for k, v in busqueda.best_params_.items()}
        filas.append({
            "modelo": nombre,
            "r2_cv_interna": round(float(busqueda.best_score_), 4),
            "r2_prueba": round(float(r2_prueba), 4),
            "mae_usd_prueba": round(mae_usd),
            "segundos": round(time.perf_counter() - t, 1),
            "mejores_parametros": json.dumps(limpio, ensure_ascii=False),
        })
        for i in range(len(busqueda.cv_results_["mean_test_score"])):
            espacio.append({
                "modelo": nombre,
                "r2_medio": round(float(busqueda.cv_results_["mean_test_score"][i]), 4),
                "r2_desv": round(float(busqueda.cv_results_["std_test_score"][i]), 4),
                "parametros": json.dumps(
                    {k.replace("modelo__", ""): (round(v, 5) if isinstance(v, float) else v)
                     for k, v in busqueda.cv_results_["params"][i].items()},
                    ensure_ascii=False),
            })

        print(f"    R² interna {busqueda.best_score_:.4f} · prueba {r2_prueba:.4f} "
              f"· {time.perf_counter() - t:.0f} s")
        print(f"    {limpio}\n")

    salida = Path(args.out_dir)
    salida.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(filas).to_csv(salida / "ajuste_resultados.csv", index=False)
    pd.DataFrame(espacio).to_csv(salida / "ajuste_espacio.csv", index=False)

    print("=" * 72)
    print(pd.DataFrame(filas)[["modelo", "r2_cv_interna", "r2_prueba",
                               "mae_usd_prueba", "segundos"]].to_string(index=False))
    print(f"\nDuración total: {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
