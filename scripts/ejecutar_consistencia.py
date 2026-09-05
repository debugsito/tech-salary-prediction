#!/usr/bin/env python3
"""Consistencia de los hallazgos centrales entre las tres ediciones.

Los resultados del Capítulo V proceden de la edición 2023. Si dependieran de
las particularidades de una recogida concreta, no describirían el fenómeno
sino el archivo. Este script reúne, para 2022, 2023 y 2025, las magnitudes
que sostienen cada hipótesis, y las presenta juntas para que la consistencia
(o su ausencia) quede medida en lugar de supuesta.

No ejecuta nada: lee los artefactos ya producidos por `ejecutar_experimento.py`
(la edición 2023 en `resultados/`, las otras en `resultados/ediciones/<año>/`),
los resúmenes SHAP de `ejecutar_analisis.py` y las auditorías de equidad.

Artefactos: resultados/consistencia_ediciones.json

Uso:
    python scripts/ejecutar_consistencia.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
RES = RAIZ / "resultados"
SALIDA = RES / "consistencia_ediciones.json"

EDICIONES = {
    "2022": RES / "ediciones" / "2022",
    "2023": RES,
    "2025": RES / "ediciones" / "2025",
}


def bloque(v: str) -> str:
    if v.startswith(("language__", "database__", "platform__", "webframe__")):
        return "Tecnologías"
    if v.startswith(("Country", "income_group", "wb_region")):
        return "Geográfico"
    if v.startswith(("YearsCode", "WorkExp", "EdLevel", "DevType", "ICorPM")):
        return "Capital humano"
    if v.startswith(("Age", "Gender")):
        return "Demográfico"
    return "Organizacional"


def resumen_edicion(anio: str, ruta: Path) -> dict:
    cv = pd.read_csv(ruta / "cv_por_codificacion.csv")
    meta = json.loads((ruta / "metadatos_ejecucion.json").read_text(encoding="utf-8"))
    # El archivo de contrastes trae dos métricas; sin filtrar, las filas de
    # MAE pisarían a las de R² en el diccionario y cambiarían los signos.
    wil = pd.read_csv(ruta / "contrastes_wilcoxon.csv")
    wil = wil[wil.metrica == "R2"]

    g = cv.groupby(["modelo", "codificacion"])["R2"].mean()
    mejor = g.idxmax()
    baselines = wil[wil.familia == "HG_vs_baseline"]
    he1 = wil[wil.familia == "HE1_codificacion"]

    # Dirección de HE1 por modelo: positiva si la codificación por objetivo gana.
    direccion_he1 = {}
    for _, f in he1.iterrows():
        modelo = f.config_a.split("/")[0]
        direccion_he1[modelo] = {
            "diferencia": round(float(f.diferencia), 4),
            "significativo": bool(f.significativo),
        }

    salida = {
        "n": int(meta["n_total"]),
        "mejor_configuracion": f"{mejor[0]}/{mejor[1]}",
        "r2_mejor": round(float(g.max()), 4),
        "r2_ridge": round(float(g.loc["Ridge"].max()), 4),
        "hg_significativos": f"{int(baselines.significativo.sum())}/{len(baselines)}",
        "he1_por_modelo": direccion_he1,
    }

    # Jerarquía SHAP por bloque, del resumen que ejecutar_analisis exporta.
    shap = RES / f"shap_resumen_XGBoost_target_{anio}.csv"
    if shap.exists():
        imp = pd.read_csv(shap)
        pes = imp.assign(b=imp.variable.map(bloque)).groupby("b").shap_medio_abs.sum()
        pes = (pes / pes.sum() * 100).round(1).sort_values(ascending=False)
        salida["pesos_shap"] = pes.to_dict()
        salida["orden_bloques"] = list(pes.index)

    # Equidad regional, de las auditorías ya calculadas.
    fair = RES / f"equidad_wb_region_{anio}.json"
    if fair.exists():
        d = json.loads(fair.read_text(encoding="utf-8"))
        salida["equidad_region"] = {
            "razon_relativa": round(d["agregados"]["razon_disparidad_relativa"], 3),
            "peor_servido": d["agregados"]["grupo_peor_servido_relativo"],
            "significativo": bool(d["contrastes"]["significativo"]),
        }
    return salida


def main() -> int:
    consolidado = {a: resumen_edicion(a, r) for a, r in EDICIONES.items()}

    # Síntesis de consistencia: qué se replica en las tres.
    hallazgos = {
        "mejor_es_ensamblado_con_objetivo": all(
            c["mejor_configuracion"].endswith("/target")
            and not c["mejor_configuracion"].startswith(("Ridge", "Baseline"))
            for c in consolidado.values()),
        "hg_todas": all(c["hg_significativos"].split("/")[0] == c["hg_significativos"].split("/")[1]
                        for c in consolidado.values()),
        "geografico_primero_nominal": all(
            c.get("orden_bloques", [""])[0] == "Geográfico"
            for c in consolidado.values()),
        "demografico_ultimo": all(
            c.get("orden_bloques", [""])[-1] == "Demográfico"
            for c in consolidado.values()),
        "region_peor_servida": sorted({c["equidad_region"]["peor_servido"]
                                       for c in consolidado.values()
                                       if "equidad_region" in c}),
        "he1_depende_del_modelo_en_todas": all(
            len({v["diferencia"] > 0 for v in c["he1_por_modelo"].values()}) > 1
            for c in consolidado.values()),
    }

    SALIDA.write_text(json.dumps({"ediciones": consolidado, "sintesis": hallazgos},
                                 ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{SALIDA.relative_to(RAIZ)} escrito\n")
    print(f"{'':14} {'n':>7} {'mejor':>16} {'R²':>7} {'Ridge':>7} {'HG':>6} "
          f"{'razón región':>13} {'peor servido'}")
    for a, c in consolidado.items():
        eq = c.get("equidad_region", {})
        print(f"edición {a:6} {c['n']:>7,} {c['mejor_configuracion']:>16} "
              f"{c['r2_mejor']:>7} {c['r2_ridge']:>7} {c['hg_significativos']:>6} "
              f"{eq.get('razon_relativa', '—'):>13} {eq.get('peor_servido', '—')}")
    print("\nSíntesis:", json.dumps(hallazgos, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
