#!/usr/bin/env python3
"""Análisis de robustez al nivel de precios: nominal frente a PPA.

El análisis principal opera en dólares nominales. Este experimento reexpresa la
compensación en dólares internacionales (paridad de poder adquisitivo, consumo
privado) y repite las tres mediciones que dependen de la escala:

1. **Las brechas descriptivas entre mercados.** Cuánto de la razón de 4.5 a 1
   entre la mediana estadounidense y la latinoamericana es nivel de precios.
2. **El peso de la geografía en el modelo.** Se reentrena la misma
   configuración sobre el mismo subconjunto con ambas variables dependientes y
   se compara la contribución SHAP del bloque geográfico.
3. **La auditoría de equidad.** Error relativo por grupo y razón de disparidad
   en ambas escalas.

Una cuarta magnitud no necesita repetirse, y conviene dejarlo dicho: la
dispersión del logaritmo del salario DENTRO de cada país es invariante ante
cualquier reescalado multiplicativo por país, de modo que el hallazgo de
heterogeneidad intrínseca (sección 5.4.6) no puede ser un artefacto del nivel
de precios por construcción.

Artefactos: resultados/ppp_comparativa.json

Uso:
    python scripts/ejecutar_ppp.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.carga_encuesta import TARGET, cargar_encuesta  # noqa: E402
from src.equidad import auditar  # noqa: E402
from src.tuberia_caracteristicas import (construir_preprocesador,  # noqa: E402
                                     nombres_de_variables, preparar_xy)
from src.metricas import calcular_metricas  # noqa: E402
from src.registro_modelos import construir_modelos  # noqa: E402

SEMILLA = 42
REFERENCIA = RAIZ / "datos" / "referencia" / "factores_ppp.csv"
SALIDA = RAIZ / "resultados" / "ppp_comparativa.json"


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


def pesos_shap(tuberia, X_ent, X_pru) -> dict[str, float]:
    """Contribución SHAP agregada por bloque, en porcentaje del total."""
    import shap

    pre = tuberia.named_steps["preprocesador"]
    nombres = nombres_de_variables(pre)
    explicador = shap.TreeExplainer(tuberia.named_steps["modelo"])
    X_t = np.asarray(pre.transform(X_pru), dtype=float)
    valores = np.abs(explicador.shap_values(X_t)).mean(axis=0)

    acumulado: dict[str, float] = {}
    for n, v in zip(nombres, valores):
        acumulado[bloque(n)] = acumulado.get(bloque(n), 0.0) + float(v)
    total = sum(acumulado.values())
    return {k: round(v / total * 100, 1) for k, v in
            sorted(acumulado.items(), key=lambda kv: -kv[1])}


def evaluar_escala(nombre, df, objetivo_log, X, A, estratos):
    """Entrena la configuración seleccionada sobre una escala y la mide."""
    y = objetivo_log.to_numpy()
    X_ent, X_pru, y_ent, y_pru, A_ent, A_pru = train_test_split(
        X, y, A, test_size=0.20, random_state=SEMILLA, stratify=estratos)

    t0 = time.perf_counter()
    tuberia = Pipeline([
        ("preprocesador", construir_preprocesador(df, "target")),
        ("modelo", construir_modelos()["XGBoost"]),
    ]).fit(X_ent, y_ent)
    pred = tuberia.predict(X_pru)
    m = calcular_metricas(y_pru, pred)

    aud_renta = auditar(y_pru, pred, A_pru["income_group"])
    aud_region = auditar(y_pru, pred, A_pru["wb_region"])
    print(f"  [{nombre}] R² = {m['R2']:.4f}   "
          f"razón renta = {aud_renta['agregados']['razon_disparidad_relativa']:.3f}   "
          f"({time.perf_counter() - t0:.0f} s)")

    return {
        "r2": round(float(m["R2"]), 4),
        "mae_escala": round(float(m["MAE_USD"])),
        "pesos_shap": pesos_shap(tuberia, X_ent, X_pru),
        "equidad_renta": {
            "error_relativo": {g: round(v["mae_relativo"], 4)
                               for g, v in aud_renta["por_grupo"].items()},
            "razon_disparidad": round(
                aud_renta["agregados"]["razon_disparidad_relativa"], 3),
            "peor_servido": aud_renta["agregados"]["grupo_peor_servido_relativo"],
        },
        "equidad_region": {
            "error_relativo": {g: round(v["mae_relativo"], 4)
                               for g, v in aud_region["por_grupo"].items()},
            "razon_disparidad": round(
                aud_region["agregados"]["razon_disparidad_relativa"], 3),
            "peor_servido": aud_region["agregados"]["grupo_peor_servido_relativo"],
        },
    }


def main() -> int:
    ppp = pd.read_csv(REFERENCIA)
    df, _ = cargar_encuesta(anio="2023")
    n_total = len(df)

    df = df.merge(ppp[["pais", "nivel_precios", "anio_ppp"]],
                  left_on="Country", right_on="pais", how="left")
    sin_factor = df[df["nivel_precios"].isna()]["Country"].value_counts()
    df = df.dropna(subset=["nivel_precios"]).reset_index(drop=True)
    print(f"Subconjunto con factor PPA: {len(df):,} de {n_total:,} "
          f"({len(df) / n_total * 100:.2f} %)")
    if len(sin_factor):
        print("Excluidos:", dict(sin_factor))

    # La compensación en dólares internacionales: dividir por el nivel de
    # precios relativo a Estados Unidos. Para EE. UU. no cambia nada.
    df["comp_ppa"] = df[TARGET] / df["nivel_precios"]

    # 1. Brechas descriptivas en ambas escalas.
    def medianas(col):
        renta = df.groupby("income_group", observed=True)[col].median()
        region = df.groupby("wb_region", observed=True)[col].median()
        usa = df.loc[df.Country == "United States of America", col].median()
        latam = df.loc[df.wb_region == "Latin America & Caribbean", col].median()
        return {
            "por_renta": {k: round(v) for k, v in renta.items()},
            "por_region": {k: round(v) for k, v in region.items()},
            "razon_usa_latam": round(float(usa / latam), 2),
            "razon_renta_alta_baja": round(
                float(renta.max() / renta.min()), 2),
        }

    descriptivo = {"nominal": medianas(TARGET), "ppa": medianas("comp_ppa")}
    print(f"\nRazón EE. UU. / América Latina: "
          f"{descriptivo['nominal']['razon_usa_latam']} nominal → "
          f"{descriptivo['ppa']['razon_usa_latam']} PPA")

    # Invariancia de la dispersión intra-país (se comprueba, no se supone).
    disp_nominal = (df.groupby("income_group", observed=True)
                    .apply(lambda g: g.groupby("Country")["salary_log"].std().median(),
                           include_groups=False))
    df["ppa_log"] = np.log1p(df["comp_ppa"])
    disp_ppa = (df.groupby("income_group", observed=True)
                .apply(lambda g: g.groupby("Country")["ppa_log"].std().median(),
                       include_groups=False))
    dif_max = float((disp_nominal - disp_ppa).abs().max())
    print(f"Dispersión intra-país: diferencia máxima entre escalas {dif_max:.4f} "
          f"(invariante, como corresponde)")

    # 2 y 3. Modelo y auditoría en ambas escalas, mismo subconjunto y partición.
    X, _, A = preparar_xy(df)
    estratos = df["income_group"].astype(str)
    print("\nEntrenando la configuración seleccionada en ambas escalas:")
    resultados = {
        "nominal": evaluar_escala("nominal", df, df["salary_log"], X, A, estratos),
        "ppa": evaluar_escala("PPA", df, df["ppa_log"], X, A, estratos),
    }

    salida = {
        "n_subconjunto": int(len(df)),
        "n_total": int(n_total),
        "paises_excluidos": {k: int(v) for k, v in sin_factor.items()},
        "fuente_ppp": "Banco Mundial, PA.NUS.PRVT.PP / PA.NUS.FCRF, año 2023",
        "descriptivo": descriptivo,
        "dispersion_intra_pais_invariante": {
            "diferencia_maxima": round(dif_max, 4),
            "nominal": {k: round(float(v), 3) for k, v in disp_nominal.items()},
        },
        "modelo": resultados,
    }
    SALIDA.write_text(json.dumps(salida, ensure_ascii=False, indent=1),
                      encoding="utf-8")
    print(f"\n{SALIDA.relative_to(RAIZ)} escrito")

    g_nom = resultados["nominal"]["pesos_shap"].get("Geográfico")
    g_ppa = resultados["ppa"]["pesos_shap"].get("Geográfico")
    print(f"\nPeso del bloque geográfico: {g_nom} % nominal → {g_ppa} % PPA")
    print(f"Razón de disparidad (renta): "
          f"{resultados['nominal']['equidad_renta']['razon_disparidad']} nominal → "
          f"{resultados['ppa']['equidad_renta']['razon_disparidad']} PPA")
    return 0


if __name__ == "__main__":
    sys.exit(main())
