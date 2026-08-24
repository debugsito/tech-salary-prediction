#!/usr/bin/env python3
"""Mitigación de la disparidad regional actuando sobre la composición de la muestra.

Tres estrategias sobre el grupo de renta del país: reponderación (pesos inversos
a la frecuencia), submuestreo y sobremuestreo. Sin SMOTE ni generación sintética:
interpolar salarios de países distintos no corresponde a ningún mercado real.

Escribe resultados/mitigacion_resultados.csv y resultados/mitigacion_por_grupo.csv.

Uso:
    python scripts/ejecutar_mitigacion.py
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

from src.carga_encuesta import cargar_encuesta  # noqa: E402
from src.equidad import auditar  # noqa: E402
from src.tuberia_caracteristicas import construir_preprocesador, preparar_xy  # noqa: E402
from src.metricas import calcular_metricas  # noqa: E402
from src.registro_modelos import construir_modelos  # noqa: E402

SEMILLA = 42
PROPORCION_PRUEBA = 0.20
GRUPO = "income_group"


def pesos_inversos(grupos: pd.Series) -> np.ndarray:
    """Peso inverso a la frecuencia del grupo, normalizado a media 1: cada grupo
    aporta la misma masa a la pérdida."""
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
    modelo = construir_modelos()["XGBoost"]
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
    metricas = calcular_metricas(y_pru, pred)
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
    ap.add_argument("--out-dir", default="resultados")
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

    # Cuánto se pierde en exactitud por lo que se gana en equidad, frente a no mitigar.
    base = res[res.estrategia == "sin_mitigacion"].iloc[0]
    print("\nCompromiso frente a la ausencia de mitigación")
    print(f"  {'estrategia':16s} {'ΔR²':>9s} {'ΔMAE USD':>10s} {'Δrazón':>9s}")
    for _, f in res[res.estrategia != "sin_mitigacion"].iterrows():
        print(f"  {f.estrategia:16s} {f.R2 - base.R2:>+9.4f} "
              f"{f.MAE_USD - base.MAE_USD:>+10,.0f} "
              f"{f.razon_disparidad_renta - base.razon_disparidad_renta:>+9.3f}")


if __name__ == "__main__":
    main()
