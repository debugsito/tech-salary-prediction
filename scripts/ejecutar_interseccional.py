#!/usr/bin/env python3
"""Análisis interseccional de la equidad: género cruzado con región (edición 2022).

Buolamwini y Gebru (2018) demuestran que las disparidades detectables al cruzar
atributos pueden ser sustancialmente mayores que las observables en cada uno por
separado. Este análisis aplica ese criterio.

Se ejecuta sobre la edición 2022 por ser la última cuya versión pública incluye
la variable de género. Los subgrupos con menos de 30 observaciones se reportan
pero se marcan como descriptivos, conforme a §3.2.6.

Salida: resultados/equidad_interseccional_2022.json y .md
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.carga_encuesta import TARGET, cargar_encuesta  # noqa: E402
from src.tuberia_caracteristicas import construir_preprocesador, preparar_xy  # noqa: E402
from src.registro_modelos import construir_modelos  # noqa: E402

SEMILLA = 42
N_MIN = 30
LATAM = {"Brazil", "Mexico", "Argentina", "Colombia", "Chile", "Peru", "Uruguay",
         "Ecuador", "Bolivia", "Paraguay", "Venezuela, Bolivarian Republic of...",
         "Costa Rica", "Panama", "Guatemala", "Cuba", "Dominican Republic",
         "El Salvador", "Honduras", "Nicaragua"}


def main():
    df, _ = cargar_encuesta(anio="2022")
    X, y, A = preparar_xy(df)
    X_ent, X_pru, y_ent, y_pru, A_ent, A_pru = train_test_split(
        X, y, A, test_size=0.2, random_state=SEMILLA, stratify=df["income_group"])

    tub = Pipeline([("preprocesador", construir_preprocesador(df, "target")),
                    ("modelo", construir_modelos()["XGBoost"])])
    tub.fit(X_ent, y_ent)
    pred = tub.predict(X_pru)

    real_usd, pred_usd = np.expm1(y_pru), np.expm1(pred)
    d = pd.DataFrame({
        "real": real_usd, "pred": pred_usd,
        "error": pred_usd - real_usd,
        "genero": A_pru["Gender"].values,
        "region": np.where(A_pru["Country"].isin(LATAM), "América Latina", "Resto del mundo"),
        "renta": A_pru["income_group"].values,
    })
    d = d[d.genero.isin(["Man", "Woman"])]
    d["genero"] = d.genero.map({"Man": "Hombres", "Woman": "Mujeres"})

    filas = []
    for (g, r), sub in d.groupby(["genero", "region"], observed=True):
        mediana = sub.real.median()
        filas.append({
            "genero": g, "region": r, "n": len(sub),
            "suficiente": bool(len(sub) >= N_MIN),
            "mediana_real": float(mediana),
            "mae": float(sub.error.abs().mean()),
            "mae_relativo": float(sub.error.abs().mean() / mediana),
            "sesgo": float(sub.error.mean()),
        })
    t = pd.DataFrame(filas)

    lineas = ["# Análisis interseccional de equidad — edición 2022", "",
              "> Generado por `scripts/ejecutar_interseccional.py`. No editar a mano.", "",
              "| Subgrupo | N | Mediana real | MAE | MAE/mediana | Sesgo |",
              "| :--- | ---: | ---: | ---: | ---: | ---: |"]
    for _, f in t.sort_values("mae_relativo", ascending=False).iterrows():
        marca = "" if f.suficiente else " *(descriptivo)*"
        lineas.append(f"| {f.genero} · {f.region}{marca} | {f.n:,} | "
                      f"\\${f.mediana_real:,.0f} | \\${f.mae:,.0f} | "
                      f"{f.mae_relativo*100:.1f} % | \\${f.sesgo:+,.0f} |")

    # Brecha de género dentro de cada región, y comparación entre regiones.
    lineas += ["", "## Brecha del error relativo por región", "",
               "| Región | Hombres | Mujeres | Diferencia | N mujeres |",
               "| :--- | ---: | ---: | ---: | ---: |"]
    resumen = {}
    for r, sub in t.groupby("region"):
        h = sub[sub.genero == "Hombres"]
        m = sub[sub.genero == "Mujeres"]
        if not len(h) or not len(m):
            continue
        hv, mv = float(h.mae_relativo.iloc[0]), float(m.mae_relativo.iloc[0])
        resumen[r] = {"hombres": hv, "mujeres": mv, "diferencia_pp": (mv - hv) * 100,
                      "n_mujeres": int(m.n.iloc[0])}
        lineas.append(f"| {r} | {hv*100:.1f} % | {mv*100:.1f} % | "
                      f"{(mv-hv)*100:+.1f} pp | {int(m.n.iloc[0]):,} |")

    salida = Path("resultados")
    (salida / "equidad_interseccional_2022.md").write_text("\n".join(lineas), encoding="utf-8")
    (salida / "equidad_interseccional_2022.json").write_text(
        json.dumps({"subgrupos": t.to_dict("records"), "por_region": resumen},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n".join(lineas))


if __name__ == "__main__":
    main()
