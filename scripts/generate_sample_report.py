#!/usr/bin/env python3
"""Genera las cifras de muestra que cita el documento de tesis.

Produce `results/sample_report.json` y `results/sample_report.md` con el efecto
de los criterios de inclusión, la composición de la muestra y la caracterización
de la variable dependiente para cada edición.

Existe para que las cifras del texto procedan de un artefacto y no de una
transcripción manual, conforme al criterio de trazabilidad de §3.6.1. Si los
criterios de inclusión cambian, basta con volver a ejecutarlo y actualizar el
documento desde su salida.

Uso:
    python scripts/generate_sample_report.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Permite ejecutar el script directamente desde la raíz del proyecto.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader_stackoverflow import CATEGORIA_AUSENTE, TARGET, cargar_encuesta  # noqa: E402

LATAM = {
    "Brazil", "Mexico", "Argentina", "Colombia", "Chile", "Peru", "Uruguay",
    "Ecuador", "Bolivia", "Paraguay", "Venezuela, Bolivarian Republic of...",
    "Costa Rica", "Panama", "Guatemala", "Cuba", "Dominican Republic",
    "El Salvador", "Honduras", "Nicaragua",
}

EEUU = "United States of America"


def caracterizar(df: pd.DataFrame, registro: pd.DataFrame) -> dict:
    lat = df[df["Country"].isin(LATAM)]
    usa = df[df["Country"] == EEUU]
    peru = df[df["Country"] == "Peru"]

    datos = {
        "n": int(len(df)),
        "n_columnas": int(df.shape[1]),
        "paises": int(df["Country"].nunique()),
        "filtros": registro.to_dict("records"),
        "compensacion": {
            "mediana": float(df[TARGET].median()),
            "media": float(df[TARGET].mean()),
            "p25": float(df[TARGET].quantile(0.25)),
            "p75": float(df[TARGET].quantile(0.75)),
            "min": float(df[TARGET].min()),
            "max": float(df[TARGET].max()),
            "asimetria": float(df[TARGET].skew()),
            "asimetria_log": float(np.log1p(df[TARGET]).skew()),
        },
        "estratos": {
            "global": {"n": int(len(df)), "mediana": float(df[TARGET].median())},
            "estados_unidos": {"n": int(len(usa)),
                               "mediana": float(usa[TARGET].median()) if len(usa) else None},
            "latam": {"n": int(len(lat)),
                      "mediana": float(lat[TARGET].median()) if len(lat) else None},
            "peru": {"n": int(len(peru)),
                     "mediana": float(peru[TARGET].median()) if len(peru) else None},
        },
        "por_nivel_renta": {
            str(g): {"n": int(len(s)), "mediana": float(s[TARGET].median())}
            for g, s in df.groupby("income_group", observed=True)
        },
        "cardinalidad": {},
        "cobertura": {},
    }
    if len(lat) and len(usa):
        datos["razon_usa_latam"] = round(
            float(usa[TARGET].median() / lat[TARGET].median()), 2)

    for col in ("Country", "DevType", "EdLevel", "OrgSize", "RemoteWork",
                "Industry", "ICorPM", "Age", "Gender", "income_group"):
        if col in df.columns:
            # La comparación con «No declarado» se hacía solo si la columna era
            # de tipo object. Desde pandas 3 las columnas de texto son de tipo
            # string, de modo que la condición nunca se cumplía: la etiqueta de
            # ausencia se contaba como una categoría más y toda variable
            # aparecía con cobertura del 100 %.
            declarado = df[col].astype("string") != CATEGORIA_AUSENTE
            datos["cardinalidad"][col] = int(df.loc[declarado, col].nunique())
            datos["cobertura"][col] = round(float(declarado.mean()) * 100, 1)

    if "LanguageHaveWorkedWith" in df.columns:
        langs = set()
        for v in df["LanguageHaveWorkedWith"].dropna():
            langs.update(x.strip() for x in str(v).split(";") if x.strip())
        datos["lenguajes_distintos"] = len(langs)
        # Distintos de los indicadores que el pipeline genera: solo se conservan
        # las tecnologías con frecuencia suficiente. El documento citaba el
        # primer número en una tabla de predictores, donde corresponde el segundo.
        # El informe carga sin expandir las tecnologías, así que los
        # indicadores se cuentan aplicando la misma expansión sobre una copia.
        from src.data_loader_stackoverflow import expandir_multivalor
        copia = df[[c for c in ("LanguageHaveWorkedWith", "DatabaseHaveWorkedWith",
                                "PlatformHaveWorkedWith") if c in df.columns]].copy()
        indicadores = {}
        for col, clave in (("LanguageHaveWorkedWith", "language"),
                           ("DatabaseHaveWorkedWith", "database"),
                           ("PlatformHaveWorkedWith", "platform")):
            if col in copia.columns:
                copia, nuevas = expandir_multivalor(copia, col)
                indicadores[clave] = len(nuevas)
                datos.setdefault("cobertura_tecnologia", {})[clave] = round(
                    float(df[col].notna().mean()) * 100, 1)
        datos["indicadores_tecnologia"] = indicadores
        datos["lenguajes_mediana_por_persona"] = float(
            df["LanguageHaveWorkedWith"].dropna().str.count(";").add(1).median())

    if "Gender" in df.columns:
        g = df[df["Gender"].isin(["Man", "Woman"])].copy()
        if len(g):
            h = g.loc[g.Gender == "Man", TARGET].median()
            m = g.loc[g.Gender == "Woman", TARGET].median()
            lat_g = g[g["Country"].isin(LATAM)]
            datos["genero"] = {
                "n_hombres": int((df.Gender == "Man").sum()),
                "n_mujeres": int((df.Gender == "Woman").sum()),
                "n_no_binario": int(df.Gender.astype(str)
                                    .str.contains("Non-binary", na=False).sum()),
                "mediana_hombres": float(h),
                "mediana_mujeres": float(m),
                "brecha_global_pct": round(float((h - m) / h * 100), 1),
                "n_mujeres_latam": int((lat_g.Gender == "Woman").sum()),
            }
            if (lat_g.Gender == "Woman").sum() >= 10:
                hl = lat_g.loc[lat_g.Gender == "Man", TARGET].median()
                ml = lat_g.loc[lat_g.Gender == "Woman", TARGET].median()
                datos["genero"]["mediana_hombres_latam"] = float(hl)
                datos["genero"]["mediana_mujeres_latam"] = float(ml)
                datos["genero"]["brecha_latam_pct"] = round(float((hl - ml) / hl * 100), 1)

    return datos


def a_markdown(todo: dict) -> str:
    l = ["# Cifras de muestra", "",
         "> Generado por `scripts/generate_sample_report.py`. **No editar a mano.**",
         "> Las cifras que el documento de tesis cita deben tomarse de aquí.", ""]

    for anio, d in todo.items():
        l += [f"## Edición {anio}", "",
              "### Efecto de los criterios de inclusión", "",
              "| Paso | Criterio | N | Descartados |", "| ---: | :--- | ---: | ---: |"]
        for f in d["filtros"]:
            l.append(f"| {f['paso']} | {f['criterio']} | {f['n']:,} | "
                     f"{f['descartados']:,} |" if f["descartados"] else
                     f"| {f['paso']} | {f['criterio']} | {f['n']:,} | — |")
        c = d["compensacion"]
        l += ["", f"**Muestra efectiva: {d['n']:,} observaciones en "
              f"{d['paises']} países.**", "",
              "### Variable dependiente", "",
              "| Estadístico | Valor |", "| :--- | ---: |",
              f"| Mediana | \\${c['mediana']:,.0f} |",
              f"| Media | \\${c['media']:,.0f} |",
              f"| Percentil 25 | \\${c['p25']:,.0f} |",
              f"| Percentil 75 | \\${c['p75']:,.0f} |",
              f"| Mínimo | \\${c['min']:,.0f} |",
              f"| Máximo | \\${c['max']:,.0f} |",
              f"| Asimetría | {c['asimetria']:.2f} |",
              f"| Asimetría tras log1p | {c['asimetria_log']:.2f} |", "",
              "### Composición", "",
              "| Ámbito | N | % | Mediana |", "| :--- | ---: | ---: | ---: |"]
        for k, nombre in [("global", "Muestra global"),
                          ("estados_unidos", "Estados Unidos"),
                          ("latam", "América Latina"), ("peru", "Perú")]:
            e = d["estratos"][k]
            med = f"\\${e['mediana']:,.0f}" if e["mediana"] else "—"
            l.append(f"| {nombre} | {e['n']:,} | {e['n']/d['n']*100:.1f} % | {med} |")
        if "razon_usa_latam" in d:
            l += ["", f"Razón entre la mediana de Estados Unidos y la de América "
                  f"Latina: **{d['razon_usa_latam']} : 1**."]
        l += ["", "### Por nivel de renta", "",
              "| Grupo | N | Mediana |", "| :--- | ---: | ---: |"]
        for g, v in sorted(d["por_nivel_renta"].items()):
            l.append(f"| {g} | {v['n']:,} | \\${v['mediana']:,.0f} |")
        if "genero" in d:
            g = d["genero"]
            l += ["", "### Género", "",
                  f"- Hombres: {g['n_hombres']:,} · Mujeres: {g['n_mujeres']:,} · "
                  f"No binarias: {g['n_no_binario']:,}",
                  f"- Mediana: hombres \\${g['mediana_hombres']:,.0f} · "
                  f"mujeres \\${g['mediana_mujeres']:,.0f} · "
                  f"brecha bruta **{g['brecha_global_pct']} %**"]
            if "brecha_latam_pct" in g:
                l.append(f"- América Latina: hombres \\${g['mediana_hombres_latam']:,.0f} · "
                         f"mujeres \\${g['mediana_mujeres_latam']:,.0f} · "
                         f"brecha bruta **{g['brecha_latam_pct']} %** "
                         f"(N mujeres = {g['n_mujeres_latam']})")
        l += ["", "### Cardinalidad y cobertura", "",
              "| Variable | Categorías | Cobertura |", "| :--- | ---: | ---: |"]
        for k, v in d["cardinalidad"].items():
            l.append(f"| `{k}` | {v} | {d['cobertura'][k]} % |")
        if "lenguajes_distintos" in d:
            l += ["", f"Lenguajes distintos declarados: **{d['lenguajes_distintos']}**. "
                  f"Mediana por respondente: **{d['lenguajes_mediana_por_persona']:.0f}**."]
        l += ["", "---", ""]
    return "\n".join(l)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anios", nargs="+", default=["2022", "2023", "2025"])
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    todo = {}
    for anio in args.anios:
        print(f"Procesando edición {anio}...")
        df, reg = cargar_encuesta(anio=anio, expandir_tecnologias=False)
        todo[anio] = caracterizar(df, reg)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "sample_report.json").write_text(
        json.dumps(todo, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "sample_report.md").write_text(a_markdown(todo), encoding="utf-8")
    print(f"\nEscritos {out}/sample_report.json y {out}/sample_report.md")


if __name__ == "__main__":
    main()
