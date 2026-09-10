#!/usr/bin/env python3
"""Evaluación de estrategias de mitigación de la disparidad regional.

La auditoría de equidad (§5.4) detecta que el modelo presta un servicio de
calidad sensiblemente inferior a las regiones de menor renta: su error relativo
es aproximadamente el doble. Este script evalúa si ese desequilibrio puede
reducirse actuando sobre la composición de la muestra de entrenamiento, y a qué
coste en exactitud global.

**Sobre el remuestreo en problemas de regresión.** Las técnicas habituales de
tratamiento del desbalance —SMOTE y sus variantes— están formuladas para
clasificación y operan sobre la variable dependiente. Aquí el desbalance no
está en la variable dependiente, que es continua, sino en la **composición
geográfica**: el 85 % de las observaciones procede de países de renta alta. Se
evalúan por tanto tres estrategias que actúan sobre esa composición:

    reponderacion   Pesos por observación inversamente proporcionales a la
                    frecuencia de su grupo. No altera los datos.
    submuestreo     Reduce los grupos mayoritarios al tamaño del menor.
    sobremuestreo   Replica observaciones de los grupos minoritarios.

Se descarta la generación sintética de observaciones (SMOTE-R y similares) por
una razón sustantiva: interpolar entre perfiles salariales de países distintos
produciría observaciones que no corresponden a ningún mercado real, y el objeto
de la auditoría es precisamente la disparidad entre mercados.

Salida:
    results/mitigacion_resultados.csv
    results/mitigacion_por_grupo.csv

Uso:
    python scripts/run_mitigacion.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader_stackoverflow import cargar_encuesta  # noqa: E402
from src.fairness import auditar  # noqa: E402
from src.feature_pipeline_so import construir_preprocesador, preparar_xy  # noqa: E402
from src.metrics import compute_metrics  # noqa: E402
from src.model_registry import build_models  # noqa: E402

SEMILLA = 42
PROPORCION_PRUEBA = 0.20
GRUPO = "income_group"


def pesos_inversos(grupos: pd.Series) -> np.ndarray:
    """Peso inversamente proporcional a la frecuencia del grupo, normalizado.

    Cada grupo aporta la misma masa total a la función de pérdida, con
    independencia de su tamaño.
    """
    frec = grupos.value_counts(normalize=True)
    w = grupos.map(lambda g: 1.0 / frec[g]).to_numpy(dtype=float)
    return w / w.mean()


def remuestrear(X, y, grupos, modo: str, rng):
    """Devuelve índices remuestreados según la estrategia indicada."""
    idx_por_grupo = {g: np.flatnonzero((grupos == g).to_numpy())
                     for g in grupos.unique()}
    tamanos = {g: len(i) for g, i in idx_por_grupo.items()}
    objetivo = min(tamanos.values()) if modo == "submuestreo" else max(tamanos.values())

    seleccion = []
    for g, idx in idx_por_grupo.items():
        if len(idx) == objetivo:
            seleccion.append(idx)
        elif len(idx) > objetivo:
            seleccion.append(rng.choice(idx, objetivo, replace=False))
        else:
            seleccion.append(rng.choice(idx, objetivo, replace=True))
    return np.concatenate(seleccion)


def evaluar(nombre, X_ent, y_ent, g_ent, X_pru, y_pru, A_pru, df, rng):
    """Ajusta el modelo bajo una estrategia y devuelve sus métricas y auditoría."""
    modelo = build_models()["XGBoost"]
    tuberia = Pipeline([
        ("preprocesador", construir_preprocesador(df, "target")),
        ("modelo", modelo),
    ])

    t = time.perf_counter()
    if nombre == "sin_mitigacion":
        tuberia.fit(X_ent, y_ent)
    elif nombre == "reponderacion":
        tuberia.fit(X_ent, y_ent, modelo__sample_weight=pesos_inversos(g_ent))
    elif nombre in ("submuestreo", "sobremuestreo"):
        idx = remuestrear(X_ent, y_ent, g_ent, nombre, rng)
        tuberia.fit(X_ent.iloc[idx], y_ent[idx])
    else:
        raise ValueError(nombre)

    pred = tuberia.predict(X_pru)
    metricas = compute_metrics(y_pru, pred)
    aud = auditar(y_pru, pred, A_pru[GRUPO])
    aud_reg = auditar(y_pru, pred, A_pru["wb_region"])

    n_ent = len(X_ent) if nombre in ("sin_mitigacion", "reponderacion") else len(idx)
    return {
        "estrategia": nombre,
        "n_entrenamiento": n_ent,
        "R2": round(metricas["R2"], 4),
        "MAE_log": round(metricas["MAE_log"], 4),
        "MAE_USD": round(metricas["MAE_USD"]),
        "MAPE": round(metricas["MAPE"], 2),
        "razon_disparidad_renta": round(aud["agregados"]["razon_disparidad_relativa"], 3),
        "razon_disparidad_region": round(aud_reg["agregados"]["razon_disparidad_relativa"], 3),
        "cv_relativo_renta": round(aud["agregados"]["coeficiente_variacion_relativo"], 3),
        "peor_grupo": aud["agregados"]["grupo_peor_servido_relativo"],
        "segundos": round(time.perf_counter() - t, 1),
    }, aud


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--anio", default="2023")
    ap.add_argument("--estrategias", nargs="+",
                    default=["sin_mitigacion", "reponderacion", "submuestreo", "sobremuestreo"])
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    rng = np.random.default_rng(SEMILLA)
    df, _ = cargar_encuesta(anio=args.anio)
    X, y, A = preparar_xy(df)

    X_ent, X_pru, y_ent, y_pru, A_ent, A_pru = train_test_split(
        X, y, A, test_size=PROPORCION_PRUEBA, random_state=SEMILLA,
        stratify=df[GRUPO])
    g_ent = A_ent[GRUPO]

    print(f"Muestra: {len(df):,} · entrenamiento {len(X_ent):,} · prueba {len(X_pru):,}")
    print("\nComposición del conjunto de entrenamiento:")
    for g, n in g_ent.value_counts().items():
        print(f"  {g[:30]:32s} {n:>6,}  ({n/len(g_ent)*100:5.1f} %)")
    print(f"  razón mayor/menor: {g_ent.value_counts().max() / g_ent.value_counts().min():.1f}\n")

    filas, por_grupo = [], []
    for est in args.estrategias:
        print(f"[{est}] ajustando...")
        fila, aud = evaluar(est, X_ent, y_ent, g_ent, X_pru, y_pru, A_pru, df, rng)
        filas.append(fila)
        for g, v in aud["por_grupo"].items():
            por_grupo.append({
                "estrategia": est, "grupo": g, "n": v["n"],
                "mae_usd": round(v["mae"]),
                "mae_relativo": round(v["mae_relativo"] * 100, 1),
                "sesgo_usd": round(v["sesgo_sistematico"]),
            })
        print(f"    R² {fila['R2']} · MAE ${fila['MAE_USD']:,} · "
              f"razón de disparidad {fila['razon_disparidad_renta']}\n")

    salida = Path(args.out_dir)
    salida.mkdir(parents=True, exist_ok=True)
    res = pd.DataFrame(filas)
    grp = pd.DataFrame(por_grupo)
    res.to_csv(salida / "mitigacion_resultados.csv", index=False)
    grp.to_csv(salida / "mitigacion_por_grupo.csv", index=False)

    print("=" * 78)
    print(res[["estrategia", "n_entrenamiento", "R2", "MAE_USD", "MAPE",
               "razon_disparidad_renta", "razon_disparidad_region"]].to_string(index=False))

    print("\nError relativo por grupo y estrategia (%)")
    piv = grp.pivot_table(index="grupo", columns="estrategia",
                          values="mae_relativo", observed=True)
    print(piv.to_string())

    # Compromiso entre exactitud y equidad: la referencia es la ausencia de
    # mitigación, y se expresa cuánto se pierde en la primera por cada punto
    # ganado en la segunda.
    base = res[res.estrategia == "sin_mitigacion"].iloc[0]
    print("\nCompromiso frente a la ausencia de mitigación")
    print(f"  {'estrategia':16s} {'ΔR²':>9s} {'ΔMAE USD':>10s} {'Δrazón':>9s}")
    for _, f in res[res.estrategia != "sin_mitigacion"].iterrows():
        print(f"  {f.estrategia:16s} {f.R2 - base.R2:>+9.4f} "
              f"{f.MAE_USD - base.MAE_USD:>+10,.0f} "
              f"{f.razon_disparidad_renta - base.razon_disparidad_renta:>+9.3f}")


if __name__ == "__main__":
    main()
