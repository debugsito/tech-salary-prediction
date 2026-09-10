#!/usr/bin/env python3
"""Análisis del modelo seleccionado: explicabilidad, equidad y efecto de la IA.

Ajusta el modelo indicado sobre el conjunto de entrenamiento, lo evalúa sobre el
de prueba y produce los artefactos que sustentan el Capítulo V:

    shap_summary_<modelo>.csv   Importancia media por variable, con intervalos
    shap_values_<modelo>.npz    Valores calculados, para verificación independiente
    fairness_<atributo>.json    Auditoría de equidad por atributo
    fairness_report.txt         Informe legible de todas las auditorías
    ia_asociacion.csv           Asociación entre adopción de IA y compensación (2025)

Uso:
    python scripts/run_analysis.py                       # edición 2023, XGBoost
    python scripts/run_analysis.py --anio 2022           # incluye género
    python scripts/run_analysis.py --anio 2025 --ia      # análisis de adopción de IA
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader_stackoverflow import TARGET, cargar_encuesta  # noqa: E402
from src.explainability import explicar  # noqa: E402
from src.fairness import auditar, resumir  # noqa: E402
from src.feature_pipeline_so import (construir_preprocesador,  # noqa: E402
                                     nombres_de_variables, preparar_xy)
from src.metrics import compute_metrics  # noqa: E402
from src.model_registry import build_models  # noqa: E402

SEMILLA = 42
PROPORCION_PRUEBA = 0.20
ESTRATO = "income_group"

# Atributos sobre los que se audita la equidad. `Country` se excluye por su
# cardinalidad: 78 grupos producirían razones de disparidad dominadas por la
# varianza muestral de los países con menos observaciones.
ATRIBUTOS_AUDITORIA = ["income_group", "wb_region", "Age", "EdLevel",
                       "OrgSize", "Gender"]

VARIABLES_IA = ["AISelect", "AIThreat", "AIAgents", "LearnCodeAI"]


def analizar_ia(df: pd.DataFrame, salida: Path) -> pd.DataFrame | None:
    """Asociación bruta entre adopción de inteligencia artificial y compensación.

    Produce la evidencia descriptiva del apartado §5.5.1. La contribución
    condicionada se obtiene de la atribución SHAP, no de aquí.
    """
    presentes = [c for c in VARIABLES_IA if c in df.columns]
    if not presentes:
        return None

    filas = []
    for col in presentes:
        sub = df[df[col].notna()]
        if not len(sub):
            continue
        for valor, g in sub.groupby(col, observed=True):
            if len(g) < 100:
                continue
            filas.append({
                "variable": col,
                "categoria": str(valor)[:80],
                "n": int(len(g)),
                "mediana_usd": float(g[TARGET].median()),
                "media_usd": float(g[TARGET].mean()),
            })

    if not filas:
        return None

    tabla = pd.DataFrame(filas)
    # Diferencia respecto de la categoría con mediana más alta de cada variable,
    # que actúa como referencia.
    tabla["diferencia_pct"] = tabla.groupby("variable")["mediana_usd"].transform(
        lambda s: (s / s.max() - 1) * 100).round(1)
    tabla = tabla.sort_values(["variable", "mediana_usd"], ascending=[True, False])
    tabla.to_csv(salida / "ia_asociacion.csv", index=False)
    return tabla


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--anio", default="2023")
    ap.add_argument("--modelo", default="XGBoost")
    ap.add_argument("--codificacion", default="target")
    ap.add_argument("--ia", action="store_true",
                    help="Analiza la asociación entre adopción de IA y compensación")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    salida = Path(args.out_dir)
    salida.mkdir(parents=True, exist_ok=True)

    print(f"Cargando la edición {args.anio}...")
    df, registro = cargar_encuesta(anio=args.anio)
    print(f"  muestra efectiva: {len(df):,} observaciones")

    X, y, A = preparar_xy(df)
    X_ent, X_pru, y_ent, y_pru, A_ent, A_pru = train_test_split(
        X, y, A, test_size=PROPORCION_PRUEBA, random_state=SEMILLA,
        stratify=df[ESTRATO])
    print(f"  entrenamiento {len(X_ent):,} · prueba {len(X_pru):,}\n")

    modelos = build_models()
    if args.modelo not in modelos:
        raise SystemExit(f"Modelo no disponible: {args.modelo}. "
                         f"Opciones: {', '.join(modelos)}")

    tuberia = Pipeline([
        ("preprocesador", construir_preprocesador(df, args.codificacion)),
        ("modelo", modelos[args.modelo]),
    ])
    print(f"Ajustando {args.modelo} con codificación «{args.codificacion}»...")
    tuberia.fit(X_ent, y_ent)
    pred = tuberia.predict(X_pru)

    metricas = compute_metrics(y_pru, pred, prefix="test_")
    print("\nDesempeño sobre el conjunto de prueba")
    for k, v in metricas.items():
        print(f"  {k:16s} {v:>12,.4f}")

    # --- Explicabilidad
    print("\nCalculando valores SHAP...")
    etiqueta = f"{args.modelo}_{args.codificacion}_{args.anio}"
    nombres = nombres_de_variables(tuberia.named_steps["preprocesador"])
    shap_info = explicar(tuberia, X_ent, X_pru, nombres=nombres,
                         directorio=str(salida), etiqueta=etiqueta,
                         semilla=SEMILLA)
    print(f"  {shap_info['estimador']} · {shap_info['n_variables']} variables")
    print("\n  Diez variables de mayor contribución media:")
    for i, r in enumerate(shap_info["top_10"], 1):
        ic = ""
        if r.get("ic_inferior") is not None:
            ic = f"  IC95 [{r['ic_inferior']:.4f}, {r['ic_superior']:.4f}]"
        print(f"   {i:>2}. {r['variable'][:38]:40s} {r['shap_medio_abs']:.4f}{ic}")
    if "spearman_shap_vs_impureza" in shap_info:
        s = shap_info["spearman_shap_vs_impureza"]
        print(f"\n  Correlación de Spearman entre SHAP e importancia por impureza: "
              f"rho = {s['rho']}  (p = {s['p_valor']:.2e}, n = {s['n_variables']})")

    (salida / f"shap_info_{etiqueta}.json").write_text(
        json.dumps(shap_info, indent=2, ensure_ascii=False), encoding="utf-8")

    # --- Equidad
    print("\nAuditoría de equidad")
    informes, auditorias = [], {}
    umbral = float(np.abs(np.expm1(y_pru) - np.expm1(pred)).mean()) * 1.25
    for atributo in ATRIBUTOS_AUDITORIA:
        if atributo not in A_pru.columns:
            continue
        aud = auditar(y_pru, pred, A_pru[atributo], umbral_zeta=umbral)
        auditorias[atributo] = aud
        informe = resumir(aud, titulo=f"{atributo} — edición {args.anio}")
        informes.append(informe)
        print()
        print(informe)
        (salida / f"fairness_{atributo}_{args.anio}.json").write_text(
            json.dumps(aud, indent=2, ensure_ascii=False), encoding="utf-8")

    (salida / f"fairness_report_{args.anio}.txt").write_text(
        "\n\n".join(informes), encoding="utf-8")

    # --- Adopción de inteligencia artificial
    if args.ia:
        print("\nAsociación bruta entre adopción de IA y compensación")
        tabla = analizar_ia(df, salida)
        if tabla is None:
            print("  La edición no incluye variables de inteligencia artificial.")
        else:
            for var, g in tabla.groupby("variable"):
                print(f"\n  {var}")
                for _, r in g.iterrows():
                    print(f"    {r.categoria[:50]:52s} N={r.n:>6,}  "
                          f"mediana ${r.mediana_usd:>9,.0f}  {r.diferencia_pct:>+6.1f} %")

    print(f"\nArtefactos escritos en {salida}")


if __name__ == "__main__":
    main()
