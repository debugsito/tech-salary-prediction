#!/usr/bin/env python3
"""Construye la tabla de referencia de países para la derivación de la región económica.

Descarga la clasificación oficial del Banco Mundial (nivel de renta y región
geográfica) y la cruza con los nombres de país que emplea la Stack Overflow
Developer Survey, que difieren en varios casos.

Salida: datos/referencia/grupos_renta_paises.csv

Se genera como artefacto versionado y no se calcula al vuelo, para que la
clasificación empleada quede fijada y sea auditable: el Banco Mundial revisa
estos grupos cada año, y un resultado que dependa de la fecha de ejecución no
sería reproducible.

Uso:
    python scripts/construir_referencia_paises.py
"""

import argparse
import csv
import json
import urllib.request
from pathlib import Path

API = "https://api.worldbank.org/v2/country?format=json&per_page=400"

# Nombres que la encuesta usa y que no coinciden con los del Banco Mundial.
# Cada equivalencia se declara aquí de forma explícita en lugar de resolverse
# con coincidencia aproximada, que podría emparejar países distintos en silencio.
ALIAS_ENCUESTA = {
    "United States of America": "United States",
    "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
    "Russian Federation": "Russian Federation",
    "Iran, Islamic Republic of...": "Iran, Islamic Rep.",
    "Venezuela, Bolivarian Republic of...": "Venezuela, RB",
    "Republic of Korea": "Korea, Rep.",
    "The former Yugoslav Republic of Macedonia": "North Macedonia",
    "Republic of Moldova": "Moldova",
    "Hong Kong (S.A.R.)": "Hong Kong SAR, China",
    "Viet Nam": "Viet Nam",
    "Syrian Arab Republic": "Syrian Arab Republic",
    "Lao People's Democratic Republic": "Lao PDR",
    "Democratic Republic of the Congo": "Congo, Dem. Rep.",
    "Congo, Republic of the...": "Congo, Rep.",
    "United Republic of Tanzania": "Tanzania",
    "Republic of North Macedonia": "North Macedonia",
    "Czech Republic": "Czechia",
    "Slovakia": "Slovak Republic",
    "Egypt": "Egypt, Arab Rep.",
    "Turkey": "Turkiye",
    "Türkiye": "Turkiye",
    "Brunei Darussalam": "Brunei Darussalam",
    "Cape Verde": "Cabo Verde",
    "Côte d'Ivoire": "Cote d'Ivoire",
    "Kyrgyzstan": "Kyrgyz Republic",
    "Saint Kitts and Nevis": "St. Kitts and Nevis",
    "Saint Lucia": "St. Lucia",
    "Saint Vincent and the Grenadines": "St. Vincent and the Grenadines",
    "Yemen": "Yemen, Rep.",
    "Bahamas": "Bahamas, The",
    "Gambia": "Gambia, The",
    "Micronesia, Federated States of...": "Micronesia, Fed. Sts.",
    "Palestine": "West Bank and Gaza",
    "Taiwan": "Taiwan",
    "South Korea": "Korea, Rep.",
    "North Korea": "Korea, Dem. People's Rep.",
    "Democratic People's Republic of Korea": "Korea, Dem. People's Rep.",
    "Libyan Arab Jamahiriya": "Libya",
    "Somalia": "Somalia, Fed. Rep.",
    "Swaziland": "Eswatini",
    "Nomadic": None,
    "Other Country (Not Listed Above)": None,
}

# Taiwan no figura en la clasificación del Banco Mundial por su estatus político.
# Se asigna manualmente y se declara aquí para que la excepción sea visible.
EXCEPCIONES = {
    "Taiwan": ("High income", "East Asia & Pacific"),
}


def descargar():
    req = urllib.request.Request(API, headers={"User-Agent": "tesis-esan/1.0"})
    with urllib.request.urlopen(req, timeout=40) as r:
        datos = json.load(r)[1]
    return {p["name"]: (p["incomeLevel"]["value"], p["region"]["value"].strip())
            for p in datos if p["region"]["value"] != "Aggregates"}


def paises_de_la_encuesta(data_dir):
    """Países presentes en las ediciones descargadas, para verificar la cobertura."""
    import pandas as pd
    vistos = set()
    for csv_path in sorted(Path(data_dir).glob("so_survey_*.csv")):
        col = pd.read_csv(csv_path, usecols=["Country"], low_memory=False)["Country"]
        vistos |= set(col.dropna().unique())
    return vistos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="datos/descargas")
    ap.add_argument("--out", default="datos/referencia/grupos_renta_paises.csv")
    args = ap.parse_args()

    wb = descargar()
    print(f"Banco Mundial: {len(wb)} países clasificados")

    try:
        encuesta = paises_de_la_encuesta(args.data_dir)
        print(f"Encuesta: {len(encuesta)} nombres de país distintos")
    except Exception as e:
        print(f"No se pudieron leer los CSV de la encuesta ({e}); se exporta solo el Banco Mundial")
        encuesta = set()

    filas, sin_resolver = [], []
    for nombre in sorted(encuesta) or sorted(wb):
        clave = ALIAS_ENCUESTA.get(nombre, nombre)
        if clave is None:
            sin_resolver.append((nombre, "sin correspondencia por definición"))
            continue
        if nombre in EXCEPCIONES:
            renta, region = EXCEPCIONES[nombre]
            origen = "excepción declarada"
        elif clave in wb:
            renta, region = wb[clave]
            origen = "Banco Mundial"
        else:
            sin_resolver.append((nombre, "no encontrado en la clasificación"))
            continue
        filas.append({
            "country_survey": nombre,
            "country_worldbank": clave,
            "income_group": renta,
            "wb_region": region,
            "source": origen,
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["country_survey", "country_worldbank",
                                          "income_group", "wb_region", "source"])
        w.writeheader()
        w.writerows(filas)

    print(f"\nEscrito {out} con {len(filas)} países.")
    if sin_resolver:
        print(f"\nSin clasificar ({len(sin_resolver)}):")
        for n, motivo in sin_resolver:
            print(f"  {n[:50]:52s} {motivo}")
        print("\nEstos quedarán en el grupo 'No clasificado' y se reportarán como tales.")


if __name__ == "__main__":
    main()
