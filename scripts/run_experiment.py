#!/usr/bin/env python3
"""Ejecuta el experimento de la tesis y exporta todos los artefactos que cita el documento.

Diseño factorial entre estrategia de representación de variables de alta
cardinalidad y arquitectura de modelado, evaluado sobre particiones idénticas
para que la comparación admita contraste pareado (§4.2.3).

Artefactos producidos en `results/`:

    cv_by_encoding.csv     Métricas por modelo, estrategia y partición
    test_by_encoding.csv   Métricas sobre el conjunto de prueba
    wilcoxon_tests.csv     Contraste de significancia entre configuraciones
    run_metadata.json      Semilla, versiones, tiempos y tamaños de partición

Uso:
    python scripts/run_experiment.py                    # ejecución completa
    python scripts/run_experiment.py --muestra 4000     # validación rápida
    python scripts/run_experiment.py --anio 2022
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader_stackoverflow import cargar_encuesta  # noqa: E402
from src.feature_pipeline_so import (construir_preprocesador,  # noqa: E402
                                     preparar_xy)
from src.metrics import compute_metrics  # noqa: E402
from src.model_registry import build_models  # noqa: E402

SEMILLA = 42
N_PARTICIONES = 5

# La validación cruzada se repite para disponer de suficientes observaciones
# pareadas en el contraste de significancia.
#
# Motivo: la prueba de rangos con signo de Wilcoxon con cinco pares tiene un
# valor p mínimo alcanzable de 0.0625, superior al nivel de significación
# adoptado. Con solo cinco particiones **ninguna hipótesis podría aceptarse**,
# con independencia de la magnitud de la diferencia observada. Con cuatro
# repeticiones se obtienen veinte observaciones pareadas, cuyo valor p mínimo
# alcanzable es de 8 × 10⁻⁶.
N_REPETICIONES = 4

PROPORCION_PRUEBA = 0.20

# La estratificación se realiza por nivel de renta del país y no de forma
# aleatoria simple: la dispersión salarial entre mercados es de tal magnitud
# (§1.1.2) que una partición aleatoria podría producir conjuntos con composición
# geográfica desigual y estimaciones de error inestables entre particiones.
ESTRATO = "income_group"


def construir_configuraciones() -> dict:
    """Modelos a evaluar, incluidas las líneas base no informativas.

    Las líneas base son necesarias para que las mejoras de desempeño sean
    interpretables en términos absolutos, y constituyen el término de
    comparación del criterio de contraste de la hipótesis general.
    """
    modelos = {
        "Baseline_media": DummyRegressor(strategy="mean"),
        "Baseline_mediana": DummyRegressor(strategy="median"),
        "Ridge": Ridge(alpha=1.0),
    }
    modelos.update(build_models())
    return modelos


def evaluar(X, y, estratos, modelos, codificacion, df_ref, n_jobs=1):
    """Validación cruzada de una estrategia de codificación sobre todos los modelos."""
    particionador = RepeatedStratifiedKFold(n_splits=N_PARTICIONES,
                                            n_repeats=N_REPETICIONES,
                                            random_state=SEMILLA)
    n_total = N_PARTICIONES * N_REPETICIONES
    filas = []

    for i, (idx_tr, idx_va) in enumerate(particionador.split(X, estratos)):
        X_tr, X_va = X.iloc[idx_tr], X.iloc[idx_va]
        y_tr, y_va = y[idx_tr], y[idx_va]

        for nombre, modelo in modelos.items():
            t0 = time.perf_counter()
            # El preprocesador se reconstruye y reajusta en cada partición: la
            # codificación por objetivo emplea la variable dependiente, y
            # ajustarla sobre el conjunto completo produciría fuga de información.
            tuberia = Pipeline([
                ("preprocesador", construir_preprocesador(df_ref, codificacion)),
                ("modelo", modelo),
            ])
            try:
                tuberia.fit(X_tr, y_tr)
                pred = tuberia.predict(X_va)
            except Exception as e:  # noqa: BLE001
                print(f"    [{nombre}/{codificacion}] partición {i}: falló — {e}")
                continue
            m = compute_metrics(y_va, pred)
            m.update({"modelo": nombre, "codificacion": codificacion,
                      "particion": i, "segundos": round(time.perf_counter() - t0, 1)})
            filas.append(m)
        if (i + 1) % N_PARTICIONES == 0:
            print(f"    repetición {(i + 1) // N_PARTICIONES}/{N_REPETICIONES} completada")

    return pd.DataFrame(filas)


def _comparar(pivote, a, b, metrica, familia):
    """Prueba de Wilcoxon pareada entre dos configuraciones."""
    from scipy.stats import wilcoxon

    if a not in pivote.columns or b not in pivote.columns:
        return None
    x, y = pivote[a], pivote[b]
    valido = x.notna() & y.notna()
    x, y = x[valido], y[valido]
    if len(x) < 6:
        return None
    dif = x - y
    if np.allclose(dif, 0):
        stat, p = np.nan, 1.0
    else:
        try:
            stat, p = wilcoxon(x, y)
        except ValueError:
            stat, p = np.nan, 1.0
    return {
        "familia": familia, "metrica": metrica,
        "config_a": f"{a[0]}/{a[1]}", "config_b": f"{b[0]}/{b[1]}",
        "n_pares": int(len(x)),
        "media_a": round(float(x.mean()), 4),
        "media_b": round(float(y.mean()), 4),
        "diferencia": round(float(dif.mean()), 4),
        "estadistico": float(stat) if stat == stat else None,
        "p_valor": float(p),
    }


def _holm(bloque: pd.DataFrame) -> pd.DataFrame:
    """Corrección de Holm-Bonferroni dentro de una familia de comparaciones."""
    bloque = bloque.sort_values("p_valor").reset_index(drop=True)
    n = len(bloque)
    bloque["umbral_holm"] = [0.05 / (n - k) for k in range(n)]
    bloque["significativo"] = bloque["p_valor"] < bloque["umbral_holm"]
    # Holm es un procedimiento secuencial: en cuanto una comparación no supera
    # su umbral, todas las posteriores se declaran no significativas.
    fallidas = bloque.index[~bloque["significativo"]]
    if len(fallidas):
        bloque.loc[fallidas.min():, "significativo"] = False
    return bloque


def contrastes(cv: pd.DataFrame, metrica: str = "R2") -> pd.DataFrame:
    """Contrastes preregistrados, agrupados por la hipótesis que responden.

    Se comparan únicamente las configuraciones que responden a una pregunta del
    estudio, y no todos los pares posibles. Contrastar todas las combinaciones
    obligaría a una corrección por comparaciones múltiples tan severa que
    ninguna diferencia real alcanzaría significación, y además incluiría
    comparaciones que no responden a ninguna hipótesis.

    La corrección de Holm-Bonferroni se aplica dentro de cada familia.
    """
    pivote = cv.pivot_table(index="particion", columns=["modelo", "codificacion"],
                            values=metrica)
    modelos = sorted({c[0] for c in pivote.columns})
    codificaciones = sorted({c[1] for c in pivote.columns})
    reales = [m for m in modelos if not m.startswith("Baseline")]
    ensamblados = [m for m in reales if m != "Ridge"]

    filas = []

    # Familia 1 — HE1: estrategia de codificación, a igualdad de modelo.
    if len(codificaciones) == 2:
        c1, c2 = codificaciones
        for m in reales:
            r = _comparar(pivote, (m, c2), (m, c1), metrica, "HE1_codificacion")
            if r:
                filas.append(r)

    # Familia 2 — HG: superioridad sobre la línea base no informativa.
    for c in codificaciones:
        for m in reales:
            for base in [b for b in modelos if b.startswith("Baseline")]:
                r = _comparar(pivote, (m, c), (base, c), metrica, "HG_vs_baseline")
                if r:
                    filas.append(r)

    # Familia 3 — modelo lineal frente a ensamblados, a igualdad de codificación.
    # Responde al criterio de Rudin (2019) recogido en §5.7.1.
    for c in codificaciones:
        for m in ensamblados:
            r = _comparar(pivote, ("Ridge", c), (m, c), metrica, "lineal_vs_ensamblado")
            if r:
                filas.append(r)

    if not filas:
        return pd.DataFrame()

    res = pd.DataFrame(filas)
    return pd.concat([_holm(b) for _, b in res.groupby("familia")],
                     ignore_index=True).sort_values(["familia", "p_valor"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--anio", default="2023")
    ap.add_argument("--muestra", type=int, default=None,
                    help="Submuestra aleatoria para validación rápida")
    ap.add_argument("--codificaciones", nargs="+", default=["target", "onehot"])
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    t_inicio = time.perf_counter()
    print(f"Cargando la edición {args.anio}...")
    df, registro = cargar_encuesta(anio=args.anio)
    print(registro.to_string(index=False))

    if args.muestra:
        df = df.sample(min(args.muestra, len(df)), random_state=SEMILLA).reset_index(drop=True)
        print(f"\nSubmuestra de validación: {len(df):,} observaciones")

    X, y, A = preparar_xy(df)
    estratos = df[ESTRATO]
    print(f"\nPredictores: {X.shape[1]} · Observaciones: {len(X):,}")

    X_ent, X_prueba, y_ent, y_prueba, e_ent, _, A_ent, A_prueba = train_test_split(
        X, y, estratos, A, test_size=PROPORCION_PRUEBA,
        random_state=SEMILLA, stratify=estratos)
    print(f"Entrenamiento: {len(X_ent):,} · Prueba: {len(X_prueba):,}")

    modelos = construir_configuraciones()
    print(f"Modelos: {', '.join(modelos)}\n")

    cv_todas, test_todas = [], []
    for codificacion in args.codificaciones:
        print(f"[{codificacion}] validación cruzada")
        cv = evaluar(X_ent, y_ent, e_ent, modelos, codificacion, df)
        cv_todas.append(cv)

        print(f"[{codificacion}] ajuste final y evaluación sobre el conjunto de prueba")
        for nombre, modelo in modelos.items():
            tuberia = Pipeline([
                ("preprocesador", construir_preprocesador(df, codificacion)),
                ("modelo", modelo),
            ])
            try:
                tuberia.fit(X_ent, y_ent)
                pred = tuberia.predict(X_prueba)
            except Exception as e:  # noqa: BLE001
                print(f"    [{nombre}] falló — {e}")
                continue
            m = compute_metrics(y_prueba, pred, prefix="test_")
            m.update({"modelo": nombre, "codificacion": codificacion})
            test_todas.append(m)
        print()

    salida = Path(args.out_dir)
    salida.mkdir(parents=True, exist_ok=True)

    cv = pd.concat(cv_todas, ignore_index=True)
    test = pd.DataFrame(test_todas)
    cv.to_csv(salida / "cv_by_encoding.csv", index=False)
    test.to_csv(salida / "test_by_encoding.csv", index=False)

    pruebas = pd.concat([contrastes(cv, m) for m in ("R2", "MAE_log")],
                        ignore_index=True)
    pruebas["p_valor"] = pruebas["p_valor"].round(8)
    pruebas["umbral_holm"] = pruebas["umbral_holm"].round(6)
    pruebas.to_csv(salida / "wilcoxon_tests.csv", index=False)

    metadatos = {
        "anio": args.anio,
        "semilla": SEMILLA,
        "n_particiones": N_PARTICIONES,
        "n_repeticiones": N_REPETICIONES,
        "n_pares_contraste": N_PARTICIONES * N_REPETICIONES,
        "estratificacion": ESTRATO,
        "n_total": int(len(df)),
        "n_entrenamiento": int(len(X_ent)),
        "n_prueba": int(len(X_prueba)),
        "n_predictores": int(X.shape[1]),
        "codificaciones": args.codificaciones,
        "modelos": list(modelos),
        "submuestra": args.muestra,
        "python": platform.python_version(),
        "duracion_segundos": round(time.perf_counter() - t_inicio, 1),
    }
    (salida / "run_metadata.json").write_text(
        json.dumps(metadatos, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 72)
    resumen = (cv.groupby(["modelo", "codificacion"])[["R2", "MAE_log", "MAE_USD"]]
               .agg(["mean", "std"]).round(4))
    print(resumen.to_string())
    print("\nArtefactos escritos en", salida)
    print(f"Duración total: {metadatos['duracion_segundos']} s")


if __name__ == "__main__":
    main()
