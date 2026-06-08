#!/usr/bin/env python3
"""Exploración de los datasets Stack Overflow Developer Survey para la tesis.

Genera el informe de viabilidad que sustenta docs/ANALISIS_DATASETS_SO.md:
cobertura por país y región, viabilidad de la auditoría de equidad,
comparabilidad entre ediciones y relación entre uso de IA y salario.

Uso:
    python scripts/explorar_ediciones.py [--data-dir datos/descargas]

Los CSV se descargan del repositorio oficial de Stack Overflow (licencia ODbL 1.0):
    https://github.com/StackExchange/Survey/raw/refs/heads/main/packages/archive/{año}/results.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "ConvertedCompYearly"

# Umbral mínimo de observaciones por grupo para considerarlo analizable.
# 30 es la convención usada en la literatura que explota esta misma encuesta.
N_MIN_GRUPO = 30

LATAM = {
    "Brazil", "Mexico", "Argentina", "Colombia", "Chile", "Peru", "Uruguay",
    "Ecuador", "Bolivia", "Paraguay", "Venezuela, Bolivarian Republic of...",
    "Costa Rica", "Panama", "Guatemala", "Cuba", "Dominican Republic",
    "El Salvador", "Honduras", "Nicaragua",
}

# Núcleo de predictores que interesa rastrear entre ediciones.
PREDICTORES = [
    "Country", TARGET, "YearsCode", "YearsCodePro", "WorkExp", "EdLevel",
    "DevType", "OrgSize", "RemoteWork", "Employment", "Age", "Gender",
    "Ethnicity", "Industry", "LanguageHaveWorkedWith", "DatabaseHaveWorkedWith",
]


def cargar(data_dir, anio):
    ruta = Path(data_dir) / f"so_survey_{anio}.csv"
    if not ruta.exists():
        print(f"  [omitido] no existe {ruta}")
        return None
    return pd.read_csv(ruta, low_memory=False)


def seccion(titulo):
    print(f"\n{'=' * 72}\n{titulo}\n{'=' * 72}")


def resumen_edicion(df, anio):
    con_salario = df[df[TARGET].notna()]
    print(f"\n{anio}: {len(df):,} respuestas × {df.shape[1]} columnas")
    print(f"  con salario: {len(con_salario):,} ({len(con_salario) / len(df) * 100:.1f}%)")
    print(f"  países con salario: {con_salario['Country'].nunique()}")
    vc = con_salario["Country"].value_counts()
    print(f"  países con N>={N_MIN_GRUPO}: {(vc >= N_MIN_GRUPO).sum()}"
          f" | N>=100: {(vc >= 100).sum()} | N>=500: {(vc >= 500).sum()}")
    return con_salario


def cobertura_latam(d, anio):
    lat = d[d["Country"].isin(LATAM)]
    print(f"\n  LATAM {anio}: {len(lat):,} con salario ({len(lat) / len(d) * 100:.1f}% del total)")
    for pais, n in lat["Country"].value_counts().items():
        mediana = lat.loc[lat.Country == pais, TARGET].median()
        aviso = "  <- por debajo del umbral" if n < N_MIN_GRUPO else ""
        print(f"    {pais[:34]:34s} {n:>5,}  mediana ${mediana:>9,.0f}{aviso}")


def auditoria_genero(d):
    """Viabilidad de la auditoría de equidad por género (solo ediciones <= 2022)."""
    if "Gender" not in d.columns:
        print("\n  Sin variable de género: Stack Overflow la retiró del dataset"
              " público a partir de 2023.")
        return

    d = d.copy()
    d["g"] = np.where(d.Gender == "Man", "Hombre",
                      np.where(d.Gender == "Woman", "Mujer", "Otro/NR"))
    d["region"] = np.where(d.Country.isin(LATAM), "LATAM", "Resto")

    print(f"\n  Cobertura de género: {d.Gender.notna().mean() * 100:.1f}%")
    print("\n  N y mediana salarial por género:")
    for g, sub in d.groupby("g"):
        print(f"    {g:10s} N={len(sub):>6,}  mediana ${sub[TARGET].median():>9,.0f}")

    binario = d[d.g.isin(["Hombre", "Mujer"])]
    print("\n  Brecha bruta por región:")
    for region, sub in binario.groupby("region"):
        h = sub.loc[sub.g == "Hombre", TARGET].median()
        m = sub.loc[sub.g == "Mujer", TARGET].median()
        n_m = (sub.g == "Mujer").sum()
        print(f"    {region:6s} hombres ${h:>9,.0f} | mujeres ${m:>9,.0f}"
              f" | brecha {(h - m) / h * 100:5.1f}%  (N mujeres = {n_m:,})")


def efecto_ia(d):
    """Asociación entre uso/percepción de IA y salario (edición 2025)."""
    for col in ["AISelect", "AIThreat", "AIAgents", "LearnCodeAI"]:
        if col not in d.columns:
            continue
        g = d.groupby(col)[TARGET].agg(["count", "median"])
        g = g[g["count"] >= 100].sort_values("median", ascending=False)
        if not len(g):
            continue
        print(f"\n  {col}:")
        for k, r in g.head(6).iterrows():
            print(f"    {str(k)[:52]:52s} N={int(r['count']):>6,}  mediana ${r['median']:>9,.0f}")


def comparabilidad(dfs):
    print(f"\n  {'variable':30s}" + "".join(f"{a:>10s}" for a in dfs))
    for col in PREDICTORES:
        fila = f"  {col:30s}"
        for df in dfs.values():
            fila += f"{f'{df[col].notna().mean() * 100:.1f}%' if col in df.columns else '—':>10s}"
        print(fila)

    anios = list(dfs)
    if len(anios) >= 2:
        a, b = anios[0], anios[-1]
        comunes = set(dfs[a].columns) & set(dfs[b].columns)
        print(f"\n  Columnas comunes entre {a} y {b}: {len(comunes)}"
              f" (de {dfs[a].shape[1]} y {dfs[b].shape[1]})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="datos/descargas")
    ap.add_argument("--anios", nargs="+", default=["2022", "2023", "2025"])
    args = ap.parse_args()

    dfs = {}
    seccion("COBERTURA POR EDICIÓN")
    for anio in args.anios:
        df = cargar(args.data_dir, anio)
        if df is None:
            continue
        dfs[anio] = df
        d = resumen_edicion(df, anio)
        cobertura_latam(d, anio)

    if not dfs:
        print("\nNo se encontró ningún CSV. Descárgalos primero (ver docs/ANALISIS_DATASETS_SO.md §1).")
        return

    seccion("VIABILIDAD DE LA AUDITORÍA DE EQUIDAD POR GÉNERO")
    for anio, df in dfs.items():
        print(f"\n{anio}:")
        auditoria_genero(df[df[TARGET].notna()])

    if "2025" in dfs:
        seccion("USO DE IA Y SALARIO (2025)")
        efecto_ia(dfs["2025"][dfs["2025"][TARGET].notna()])

    seccion("COMPARABILIDAD ENTRE EDICIONES (% no nulo)")
    comparabilidad(dfs)


if __name__ == "__main__":
    main()
