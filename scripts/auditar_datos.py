#!/usr/bin/env python3
"""Auditoría de la calidad del conjunto de datos tras la limpieza.

Comprueba supuestos que la cadena de análisis da por buenos y que, de no
cumplirse, invalidarían resultados: duplicados, fuga de información entre
particiones, colinealidad, categorías con soporte insuficiente, coherencia
interna de las variables y valores límite.

Uso:
    python scripts/auditar_datos.py --anio 2023
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.carga_encuesta import TARGET, cargar_encuesta  # noqa: E402
from src.tuberia_caracteristicas import (CAT_ALTA, CAT_BAJA, NUMERICAS,  # noqa: E402
                                     columnas_disponibles, columnas_tecnologia)

def seccion(t): print(f"\n{'─'*72}\n{t}\n{'─'*72}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anio", default="2023")
    args = ap.parse_args()

    df, reg = cargar_encuesta(anio=args.anio)
    print(f"Edición {args.anio} · muestra efectiva {len(df):,} × {df.shape[1]}")

    problemas = []

    seccion("1. Duplicados")
    print(f"  Filas idénticas en su totalidad: {df.duplicated().sum():,}")
    # Duplicados sobre los predictores: dos personas con perfil idéntico y
    # salario distinto no son un error, pero si son muchos indicarían que el
    # conjunto de predictores no discrimina lo suficiente.
    pred = (columnas_disponibles(df, NUMERICAS) + columnas_disponibles(df, CAT_BAJA)
            + columnas_disponibles(df, CAT_ALTA) + columnas_tecnologia(df))
    dup_pred = df.duplicated(subset=pred).sum()
    print(f"  Perfiles de predictores repetidos: {dup_pred:,} ({dup_pred/len(df)*100:.2f} %)")
    if dup_pred / len(df) > 0.05:
        problemas.append(f"{dup_pred/len(df)*100:.1f} % de perfiles repetidos")
    if "ResponseId" in df.columns:
        print(f"  Identificadores de respuesta repetidos: {df.ResponseId.duplicated().sum()}")

    seccion("2. Valores faltantes tras la limpieza")
    faltan = df[pred].isna().mean().sort_values(ascending=False)
    faltan = faltan[faltan > 0]
    if len(faltan):
        for c, v in faltan.head(10).items():
            print(f"  {c[:44]:46s} {v*100:5.1f} %")
    else:
        print("  Ninguno en los predictores.")
    print(f"  Variable dependiente: {df[TARGET].isna().sum()} nulos")

    seccion("3. Colinealidad entre las variables continuas")
    num = columnas_disponibles(df, NUMERICAS)
    corr = df[num].corr()
    for i, a in enumerate(num):
        for b in num[i+1:]:
            r = corr.loc[a, b]
            marca = "  ← elevada" if abs(r) > 0.8 else ""
            print(f"  {a[:22]:24s} ↔ {b[:22]:24s} r = {r:+.3f}{marca}")
            if abs(r) > 0.8:
                problemas.append(f"colinealidad {a}↔{b} r={r:.2f}")

    seccion("4. Categorías con soporte insuficiente")
    # El codificador disyuntivo agrupa por debajo de 30; conviene saber cuántas
    # categorías caen en ese grupo residual y qué proporción de la muestra suponen.
    for c in columnas_disponibles(df, CAT_BAJA + CAT_ALTA):
        vc = df[c].value_counts()
        raras = vc[vc < 30]
        if len(raras):
            print(f"  {c[:20]:22s} {len(raras):>3} de {len(vc):>3} categorías "
                  f"con N<30, que suman {raras.sum():,} filas ({raras.sum()/len(df)*100:.1f} %)")

    seccion("5. Coherencia interna")
    if {"YearsCode_num", "YearsCodePro_num"} <= set(df.columns):
        m = df[["YearsCode_num", "YearsCodePro_num"]].dropna()
        inc = (m.YearsCodePro_num > m.YearsCode_num).sum()
        print(f"  Experiencia profesional mayor que años programando: {inc:,} "
              f"({inc/len(m)*100:.2f} %)")
        if inc / len(m) > 0.02:
            problemas.append(f"{inc:,} filas con experiencia profesional > años programando")
    if "Age" in df.columns and "YearsCode_num" in df.columns:
        # Edad mínima del tramo frente a años programando: detecta declaraciones
        # imposibles, como programar desde antes de nacer.
        minimos = {"Under 18 years old": 18, "18-24 years old": 18, "25-34 years old": 25,
                   "35-44 years old": 35, "45-54 years old": 45, "55-64 years old": 55,
                   "65 years or older": 65}
        e = df["Age"].map(minimos)
        imp = ((df.YearsCode_num > e - 5) & e.notna()).sum()
        print(f"  Años programando incompatibles con el tramo de edad: {imp:,} "
              f"({imp/len(df)*100:.2f} %)")
        if imp / len(df) > 0.01:
            problemas.append(f"{imp:,} filas con años programando incompatibles con la edad")

    seccion("6. Distribución de la variable dependiente")
    s = df[TARGET]
    print(f"  Mínimo ${s.min():,.0f} · P1 ${s.quantile(.01):,.0f} · "
          f"P99 ${s.quantile(.99):,.0f} · Máximo ${s.max():,.0f}")
    print(f"  Asimetría {s.skew():.2f} · curtosis {s.kurtosis():.2f}")
    print(f"  Tras logaritmo: asimetría {df.salary_log.skew():.2f} · "
          f"curtosis {df.salary_log.kurtosis():.2f}")
    bajo = (s < 5000).sum()
    print(f"  Por debajo de $5,000 anuales: {bajo:,} ({bajo/len(df)*100:.2f} %)")

    seccion("7. Desbalance de representación")
    for c in ("income_group", "wb_region"):
        if c in df.columns:
            vc = df[c].value_counts(normalize=True)
            print(f"\n  {c}")
            for k, v in vc.items():
                print(f"    {str(k)[:38]:40s} {v*100:5.1f} %")
            print(f"    razón mayor/menor: {vc.max()/vc.min():.1f}")

    seccion("8. Riesgo de fuga por la codificación por objetivo")
    # La codificación por objetivo usa la variable dependiente. Si una categoría
    # tiene muy pocas observaciones, su media condicionada se aproxima al valor
    # individual y el suavizado deja de protegerla.
    for c in columnas_disponibles(df, CAT_ALTA):
        vc = df[c].value_counts()
        criticas = vc[vc < 10]
        print(f"  {c}: {len(criticas)} categorías con N<10 "
              f"({criticas.sum()} filas). Suavizado actual: 10.0")

    seccion("Resumen")
    if problemas:
        print(f"  {len(problemas)} cuestiones a revisar:")
        for p in problemas:
            print(f"    · {p}")
    else:
        print("  Sin cuestiones bloqueantes.")


if __name__ == "__main__":
    main()
