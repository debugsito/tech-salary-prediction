#!/usr/bin/env python3
"""Transferencia del modelo desplegado (servicio/artefactos) a los salarios de
aijobs.net (CC0), una fuente con esquema mínimo: país, rol, modalidad y experiencia por tramos.

Como control se evalúa el mismo modelo sobre su propia prueba enmascarada al mismo
esquema, para separar el coste de las variables ausentes del cambio de instrumento.
La conversión de tramos a años se prueba con dos anclas.
Escribe resultados/transferencia_aijobs.json.

Uso:
    python scripts/ejecutar_transferencia.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.carga_encuesta import TARGET, cargar_encuesta  # noqa: E402
from src.tuberia_caracteristicas import preparar_xy  # noqa: E402
from src.metricas import calcular_metricas  # noqa: E402

SEMILLA = 42
CSV = RAIZ / "datos" / "descargas" / "aijobs_salaries.csv"
MODELO = RAIZ / "servicio" / "artefactos" / "modelo.joblib"
CONTEXTO = RAIZ / "servicio" / "artefactos" / "contexto.json"
SALIDA = RAIZ / "resultados" / "transferencia_aijobs.json"

# Rol del portal → rol de la encuesta 2023. Solo correspondencias defendibles;
# el resto queda fuera.
MAPEO_ROL = {
    "Data Scientist": "Data scientist or machine learning specialist",
    "Applied Scientist": "Data scientist or machine learning specialist",
    "Research Scientist": "Data scientist or machine learning specialist",
    "Machine Learning Engineer": "Data scientist or machine learning specialist",
    "ML Engineer": "Data scientist or machine learning specialist",
    "Research Engineer": "Data scientist or machine learning specialist",
    "Data Engineer": "Engineer, data",
    "Analytics Engineer": "Engineer, data",
    "Big Data Engineer": "Engineer, data",
    "Data Analyst": "Data or business analyst",
    "Business Intelligence Analyst": "Data or business analyst",
    "BI Analyst": "Data or business analyst",
}

# ISO del portal → nombre de país de la encuesta, solo los presentes en la muestra.
MAPEO_PAIS = {
    "US": "United States of America",
    "GB": "United Kingdom of Great Britain and Northern Ireland",
    "CA": "Canada", "ES": "Spain", "DE": "Germany", "IN": "India",
    "FR": "France", "AU": "Australia", "CO": "Colombia", "PT": "Portugal",
    "NL": "Netherlands", "BR": "Brazil", "MX": "Mexico", "IT": "Italy",
    "PL": "Poland", "IE": "Ireland", "CH": "Switzerland", "AT": "Austria",
    "GR": "Greece", "LT": "Lithuania", "UA": "Ukraine", "AR": "Argentina",
    "JP": "Japan", "SG": "Singapore", "SE": "Sweden", "DK": "Denmark",
    "FI": "Finland", "NO": "Norway", "BE": "Belgium", "CZ": "Czech Republic",
    "RO": "Romania", "TR": "Turkey", "NG": "Nigeria", "PK": "Pakistan",
}

MODALIDAD = {"0": "In-person", "50": "Hybrid (some remote, some in-person)",
             "100": "Remote"}

# Años de experiencia por tramo; la alternativa mide la sensibilidad a la conversión.
ANCLAS = {"principal": {"EN": 2, "MI": 5, "SE": 9, "EX": 15},
          "alternativa": {"EN": 1, "MI": 4, "SE": 8, "EX": 12}}

COLUMNAS_MINIMAS = {"Country", "DevType", "RemoteWork", "YearsCodePro_num",
                    "income_group"}


def fila_minima(columnas, tecnologia, pais, rol, modalidad, anios, renta):
    fila = {}
    for c in columnas:
        if c in tecnologia:
            fila[c] = 0
        elif c in ("YearsCode_num", "YearsCodePro_num", "WorkExp_num"):
            fila[c] = np.nan
        else:
            fila[c] = "No declarado"
    fila.update({"Country": pais, "DevType": rol, "RemoteWork": modalidad,
                 "YearsCodePro_num": anios, "YearsCode_num": anios,
                 "income_group": renta})
    return fila


def main() -> int:
    tuberia = joblib.load(MODELO)
    ctx = json.loads(CONTEXTO.read_text(encoding="utf-8"))
    columnas, tecnologia = ctx["columnas"], set(ctx["tecnologia"])
    renta_de = ctx["referencia"]["income_group"]

    bruto = pd.read_csv(CSV, dtype=str)
    n0 = len(bruto)
    d = bruto[(bruto.work_year == "2023") & (bruto.employment_type == "FT")].copy()
    n1 = len(d)
    d = d[d.job_title.isin(MAPEO_ROL) & d.employee_residence.isin(MAPEO_PAIS)].copy()
    d["pais"] = d.employee_residence.map(MAPEO_PAIS)
    d = d[d.pais.isin(renta_de)].copy()
    d["salario"] = pd.to_numeric(d.salary_in_usd)
    # Mismo criterio de plausibilidad del estudio, relativo a la mediana por país
    # de la propia fuente.
    med = d.groupby("pais").salario.transform("median")
    d = d[(d.salario >= med * 0.2) & (d.salario <= med * 6)].copy()
    print(f"aijobs.net: {n0:,} filas → {n1:,} de 2023 a tiempo completo → "
          f"{len(d):,} retenidas ({len(d) / n1 * 100:.0f} % de las de 2023)")

    resultados = {}
    for nombre, ancla in ANCLAS.items():
        filas = [fila_minima(columnas, tecnologia, f.pais, MAPEO_ROL[f.job_title],
                             MODALIDAD.get(f.remote_ratio, "No declarado"),
                             ancla[f.experience_level], renta_de[f.pais])
                 for f in d.itertuples()]
        X_ext = pd.DataFrame(filas, columns=columnas)
        pred = tuberia.predict(X_ext)
        m = calcular_metricas(np.log1p(d.salario.to_numpy()), pred)
        resultados[nombre] = {k: round(float(v), 4) for k, v in m.items()}
        print(f"  ancla {nombre}: R² = {m['R2']:.4f}   MAE = ${m['MAE_USD']:,.0f}"
              f"   MAPE = {m['MAPE']:.1f} %")

    # Controles sobre el instrumento propio: prueba completa y enmascarada.
    df, _ = cargar_encuesta(anio="2023")
    X, _, _ = preparar_xy(df)
    y = df["salary_log"].to_numpy()
    estratos = df["income_group"].astype(str)
    _, X_pru, _, y_pru = train_test_split(X, y, test_size=0.20,
                                          random_state=SEMILLA, stratify=estratos)
    completa = calcular_metricas(y_pru, tuberia.predict(X_pru))

    X_masc = X_pru.copy()
    for c in X_masc.columns:
        if c in COLUMNAS_MINIMAS:
            continue
        if c in tecnologia:
            X_masc[c] = 0
        elif c in ("YearsCode_num", "WorkExp_num"):
            X_masc[c] = X_pru["YearsCodePro_num"]
        else:
            X_masc[c] = "No declarado"
    enmascarada = calcular_metricas(y_pru, tuberia.predict(X_masc))
    print(f"\nControl, prueba propia completa:    R² = {completa['R2']:.4f}"
          f"   MAE = ${completa['MAE_USD']:,.0f}")
    print(f"Control, prueba propia enmascarada: R² = {enmascarada['R2']:.4f}"
          f"   MAE = ${enmascarada['MAE_USD']:,.0f}")

    # Calibración global y por rol con el ancla principal. Las varianzas explican
    # por qué el R² no es comparable entre poblaciones.
    ancla = ANCLAS["principal"]
    filas = [fila_minima(columnas, tecnologia, f.pais, MAPEO_ROL[f.job_title],
                         MODALIDAD.get(f.remote_ratio, "No declarado"),
                         ancla[f.experience_level], renta_de[f.pais])
             for f in d.itertuples()]
    pred_usd = np.expm1(tuberia.predict(pd.DataFrame(filas, columns=columnas)))
    real_usd = d.salario.to_numpy()
    d2 = d.assign(rol=d.job_title.map(MAPEO_ROL), pred=pred_usd)
    med_rol = {}
    for rol, g in d2.groupby("rol"):
        med_rol[rol] = {
            "n": int(len(g)),
            "mediana_fuente": round(float(g.salario.median())),
            "mediana_predicha": round(float(g.pred.median())),
            "razon": round(float(g.pred.median() / g.salario.median()), 3),
        }
    sel_usa = (d2.pais == "United States of America").to_numpy()
    solo_usa = calcular_metricas(np.log1p(real_usd[sel_usa]),
                               np.log1p(pred_usd[sel_usa]))
    diagnostico = {
        "mediana_real": round(float(np.median(real_usd))),
        "mediana_predicha": round(float(np.median(pred_usd))),
        "razon_global": round(float(np.median(pred_usd) / np.median(real_usd)), 3),
        "var_log_externa": round(float(np.var(np.log1p(real_usd))), 3),
        "var_log_prueba_propia": round(float(np.var(y_pru)), 3),
        "r2_solo_usa": round(float(solo_usa["R2"]), 3),
        "mape_solo_usa": round(float(solo_usa["MAPE"]), 1),
    }
    SALIDA.write_text(json.dumps({
        "fuente": "aijobs.net salaries (CC0), año 2023, tiempo completo",
        "n_descargadas": int(n0), "n_2023_ft": int(n1), "n_retenidas": int(len(d)),
        "paises": int(d.pais.nunique()),
        "fraccion_usa": round(float((d.pais == "United States of America").mean()), 3),
        "anclas_experiencia": ANCLAS,
        "externa": resultados,
        "control_completa": {k: round(float(v), 4) for k, v in completa.items()},
        "control_enmascarada": {k: round(float(v), 4) for k, v in enmascarada.items()},
        "diagnostico": diagnostico,
        "por_rol": med_rol,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n{SALIDA.relative_to(RAIZ)} escrito")
    return 0


if __name__ == "__main__":
    sys.exit(main())
