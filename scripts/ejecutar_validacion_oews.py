#!/usr/bin/env python3
"""Validación externa frente a la estadística oficial estadounidense (OEWS).

La sección 3.2.3 del documento declara que la muestra no es probabilística, y
hasta ahora esa limitación se resolvía con una advertencia. Este experimento la
convierte en una medición: contrasta las medianas de la muestra y las
estimaciones del modelo con las de la Occupational Employment and Wage
Statistics (BLS), una encuesta a establecimientos con diseño probabilístico,
sobre el único país donde ambas fuentes tienen profundidad suficiente.

Se emparejan los años: edición 2025 de la encuesta frente a las estimaciones
de mayo de 2025 de OEWS, que son las únicas que la API pública del BLS sirve
(el servidor de archivos históricos rechaza las descargas automatizadas, y la
API solo publica la referencia vigente). Las medianas oficiales se fijan en
`datos/referencia/oews_medianas_2025.json` para que el resultado no dependa de
la fecha de consulta.

El mapeo entre roles de la encuesta y ocupaciones SOC es imperfecto y se
declara por completo: cinco roles de desarrollo comparten la ocupación
15-1252, y los roles sin correspondencia defendible (DevOps, arquitectura,
ingeniería de datos, IA/ML, infraestructura en la nube) quedan fuera y se
listan. Las definiciones también difieren: la encuesta pregunta compensación
total autorreportada; OEWS mide el salario que paga el establecimiento. El
contraste informa de posición y orden, no de una igualdad esperada.

Artefactos: resultados/validacion_oews.json

Uso:
    python scripts/ejecutar_validacion_oews.py
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.carga_encuesta import TARGET, cargar_encuesta  # noqa: E402
from src.tuberia_caracteristicas import construir_preprocesador, preparar_xy  # noqa: E402
from src.registro_modelos import construir_modelos  # noqa: E402

SEMILLA = 42
ANIO = "2025"
REFERENCIA = RAIZ / "datos" / "referencia" / "oews_medianas_2025.json"
SALIDA = RAIZ / "resultados" / "validacion_oews.json"
API = "https://api.bls.gov/publicAPI/v2/timeseries/data/"

# Ocupaciones SOC 2018 comparadas. La serie OEWS se compone como
# OEU + N (nacional) + área 0000000 + industria 000000 + ocupación + 04
# (mediana salarial anual).
OCUPACIONES = {
    "15-1252": "Software Developers",
    "15-1254": "Web Developers",
    "15-1253": "Software Quality Assurance Analysts and Testers",
    "15-2051": "Data Scientists",
    "11-3021": "Computer and Information Systems Managers",
}

# Rol de la edición 2025 → ocupación SOC. Los roles de desarrollo de
# aplicaciones comparten la 15-1252, que es como el SOC los clasifica.
MAPEO_SOC = {
    "Developer, full-stack": "15-1252",
    "Developer, back-end": "15-1252",
    "Developer, desktop or enterprise applications": "15-1252",
    "Developer, embedded applications or devices": "15-1252",
    "Developer, mobile": "15-1252",
    "Developer, front-end": "15-1254",
    "Developer, QA or test": "15-1253",
    "Data scientist": "15-2051",
    "Engineering manager": "11-3021",
}

PAIS = "United States of America"
N_MINIMO = 30


def medianas_oficiales() -> dict[str, int]:
    """Medianas anuales de OEWS mayo 2025, de la referencia o de la API."""
    if REFERENCIA.exists():
        return json.loads(REFERENCIA.read_text(encoding="utf-8"))["medianas"]

    series = {f"OEUN0000000000000{soc.replace('-', '')}04": soc
              for soc in OCUPACIONES}
    cuerpo = json.dumps({"seriesid": list(series), "startyear": "2025",
                         "endyear": "2025"}).encode()
    req = urllib.request.Request(API, data=cuerpo,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as r:
        d = json.load(r)
    if d.get("status") != "REQUEST_SUCCEEDED":
        raise SystemExit(f"API del BLS: {d.get('message')}")

    medianas = {}
    for s in d["Results"]["series"]:
        datos = s.get("datos", [])
        if datos:
            medianas[series[s["seriesID"]]] = int(datos[0]["value"])
    REFERENCIA.write_text(json.dumps({
        "fuente": "BLS OEWS, nacional, mayo 2025, mediana salarial anual",
        "api": API, "series": {v: k for k, v in series.items()},
        "medianas": medianas,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{REFERENCIA.relative_to(RAIZ)} escrito: {medianas}")
    return medianas


def main() -> int:
    oficiales = medianas_oficiales()

    df, _ = cargar_encuesta(anio=ANIO)
    X, _, _ = preparar_xy(df)
    y = df["salary_log"].to_numpy()
    estratos = df["income_group"].astype(str)
    X_ent, X_pru, y_ent, y_pru = train_test_split(
        X, y, test_size=0.20, random_state=SEMILLA, stratify=estratos)

    tuberia = Pipeline([
        ("preprocesador", construir_preprocesador(df, "target")),
        ("modelo", construir_modelos()["XGBoost"]),
    ]).fit(X_ent, y_ent)
    pred = np.expm1(tuberia.predict(X_pru))

    usa = df[df["Country"] == PAIS]
    es_usa_pru = (df.loc[X_pru.index, "Country"] == PAIS).to_numpy()
    rol_pru = df.loc[X_pru.index, "DevType"].to_numpy()

    filas, sin_mapeo = [], []
    for rol, n in usa["DevType"].value_counts().items():
        if rol == "No declarado" or n < N_MINIMO:
            continue
        if rol not in MAPEO_SOC:
            sin_mapeo.append({"rol": rol, "n": int(n)})
            continue
        soc = MAPEO_SOC[rol]
        if soc not in oficiales:
            continue
        oficial = oficiales[soc]
        observada = float(usa.loc[usa["DevType"] == rol, TARGET].median())
        sel = es_usa_pru & (rol_pru == rol)
        predicha = float(np.median(pred[sel])) if sel.sum() >= N_MINIMO else None
        filas.append({
            "rol": rol, "soc": soc, "ocupacion": OCUPACIONES[soc],
            "n_muestra_usa": int(n), "n_prueba_usa": int(sel.sum()),
            "mediana_oficial": oficial,
            "mediana_muestra": round(observada),
            "razon_muestra_oficial": round(observada / oficial, 3),
            "mediana_predicha": round(predicha) if predicha else None,
            "razon_predicha_oficial": round(predicha / oficial, 3) if predicha else None,
        })

    razones_m = [f["razon_muestra_oficial"] for f in filas]
    razones_p = [f["razon_predicha_oficial"] for f in filas
                 if f["razon_predicha_oficial"]]
    resumen = {
        "fuente": "BLS OEWS, nacional, mayo 2025 (mediana salarial anual, vía API)",
        "edicion_encuesta": ANIO, "pais": PAIS,
        "n_usa": int(len(usa)),
        "n_roles_comparados": len(filas),
        "mediana_razon_muestra_oficial": round(float(np.median(razones_m)), 3),
        "rango_razon_muestra_oficial": [round(min(razones_m), 3),
                                        round(max(razones_m), 3)],
        "mediana_razon_predicha_oficial": (round(float(np.median(razones_p)), 3)
                                           if razones_p else None),
        "roles_sin_mapeo_defendible": sin_mapeo,
        "comparaciones": filas,
    }
    SALIDA.write_text(json.dumps(resumen, ensure_ascii=False, indent=1),
                      encoding="utf-8")

    print(f"\n{SALIDA.relative_to(RAIZ)}: {len(filas)} roles\n")
    print(f"{'rol':46} {'n':>5} {'oficial':>9} {'muestra':>9} {'m/of':>6} {'predicha':>9} {'p/of':>6}")
    for f in filas:
        pr = f"{f['mediana_predicha']:>9,}" if f["mediana_predicha"] else f"{'—':>9}"
        rp = f"{f['razon_predicha_oficial']:>6}" if f["razon_predicha_oficial"] else f"{'—':>6}"
        print(f"{f['rol'][:44]:46} {f['n_muestra_usa']:>5} {f['mediana_oficial']:>9,} "
              f"{f['mediana_muestra']:>9,} {f['razon_muestra_oficial']:>6} {pr} {rp}")
    print(f"\nMediana muestra/oficial:  {resumen['mediana_razon_muestra_oficial']}"
          f"   rango {resumen['rango_razon_muestra_oficial']}")
    print(f"Mediana predicha/oficial: {resumen['mediana_razon_predicha_oficial']}")
    if sin_mapeo:
        print("Sin mapeo defendible: "
              + ", ".join(f"{r['rol']} ({r['n']})" for r in sin_mapeo))
    return 0


if __name__ == "__main__":
    sys.exit(main())
