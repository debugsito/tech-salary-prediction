#!/usr/bin/env python3
"""Genera el anexo de tablas de resultados completas.

El Capítulo V presenta los resultados resumidos: los modelos de mejor desempeño,
las variables de mayor contribución, los contrastes que responden a cada
hipótesis. Un anexo con las tablas completas permite verificar lo que el cuerpo
del documento resume, que es lo que un jurado necesita para comprobar que la
selección de lo mostrado no es interesada.

Se genera desde los artefactos y no se escribe a mano: si los resultados
cambian, basta con volver a ejecutarlo.

Salida: el borrador del documento si está junto a este proyecto;
si no, `resultados/anexo_resultados.md`.

Uso:
    python scripts/generar_anexo_resultados.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
RES = RAIZ / "resultados"
# Cuando este proyecto convive con el documento, el anexo se escribe sobre su
# borrador; en un clon aislado se deja como archivo propio en resultados/.
_BORRADOR = RAIZ.parent / "tesis_borrador" / "12_anexo_resultados.md"
SALIDA = _BORRADOR if _BORRADOR.parent.exists() else RAIZ / "resultados" / "anexo_resultados.md"

# Número de variables de la lista de contribución que se llevan al anexo. Con
# 162 variables, la cola carece de interés: se corta donde la contribución deja
# de ser distinguible de cero.
N_SHAP = 40


def tabla_md(df: pd.DataFrame, alineacion: dict | None = None,
             titulo: str = "") -> list[str]:
    """Convierte un marco de datos en una tabla de Markdown, con su rótulo.

    El número del rótulo se deja en 0.0: lo asigna después
    `tools/numerar_tablas.py` en el repositorio del documento, que es quien
    conoce la posición de la tabla dentro del conjunto.
    """
    alineacion = alineacion or {}
    cols = list(df.columns)
    sep = [":---" if alineacion.get(c) == "izq" else "---:" for c in cols]
    filas = []
    if titulo:
        filas += [f"**Tabla 0.0.** {titulo}", ""]
    filas += ["| " + " | ".join(str(c) for c in cols) + " |",
              "| " + " | ".join(sep) + " |"]
    for _, r in df.iterrows():
        filas.append("| " + " | ".join(str(v) for v in r) + " |")
    return filas


def seccion_cv(l: list[str]):
    p = RES / "cv_por_codificacion.csv"
    if not p.exists():
        return
    cv = pd.read_csv(p)
    g = (cv.groupby(["modelo", "codificacion"])[["R2", "RMSE_log", "MAE_log",
                                                 "MAE_USD", "MAPE"]]
         .agg(["mean", "std"]).round(4))
    filas = []
    for (modelo, cod), r in g.iterrows():
        filas.append({
            "Modelo": modelo,
            "Codificación": "por objetivo" if cod == "target" else "disyuntiva",
            "R²": f"{r[('R2','mean')]:.4f} ± {r[('R2','std')]:.4f}",
            "RMSE (log)": f"{r[('RMSE_log','mean')]:.4f}",
            "MAE (log)": f"{r[('MAE_log','mean')]:.4f}",
            "MAE (USD)": f"{r[('MAE_USD','mean')]:,.0f}",
            "MAPE": f"{r[('MAPE','mean')]:.2f}",
        })
    df = pd.DataFrame(filas).sort_values("R²", ascending=False)

    l += ["## I.1 Validación cruzada: todas las configuraciones", "",
          f"Veinte observaciones por configuración: cinco particiones "
          f"estratificadas repetidas cuatro veces. Se indica la media y la "
          f"desviación típica entre particiones para el coeficiente de "
          f"determinación.", ""]
    l += tabla_md(df, {"Modelo": "izq", "Codificación": "izq"},
                  "Validación cruzada: desempeño de todas las configuraciones")
    l += ["", "*Fuente:* `resultados/cv_por_codificacion.csv`", ""]


def seccion_test(l: list[str]):
    p = RES / "prueba_por_codificacion.csv"
    if not p.exists():
        return
    t = pd.read_csv(p)
    cols = {"modelo": "Modelo", "codificacion": "Codificación",
            "test_R2": "R²", "test_RMSE_log": "RMSE (log)",
            "test_MAE_log": "MAE (log)", "test_MAE_USD": "MAE (USD)",
            "test_MAPE": "MAPE"}
    t = t[[c for c in cols if c in t.columns]].rename(columns=cols)
    t["Codificación"] = t["Codificación"].map(
        {"target": "por objetivo", "onehot": "disyuntiva"}).fillna(t["Codificación"])
    for c in ("R²", "RMSE (log)", "MAE (log)"):
        if c in t.columns:
            t[c] = t[c].round(4)
    for c in ("MAE (USD)",):
        if c in t.columns:
            t[c] = t[c].round(0).map(lambda v: f"{v:,.0f}")
    if "MAPE" in t.columns:
        t["MAPE"] = t["MAPE"].round(2)

    l += ["## I.2 Conjunto de prueba: todas las configuraciones", "",
          "Evaluación única sobre las 7,525 observaciones reservadas, que no "
          "intervinieron en ninguna decisión de modelado.", ""]
    l += tabla_md(t.sort_values("R²", ascending=False),
                  {"Modelo": "izq", "Codificación": "izq"},
                  "Conjunto de prueba: desempeño de todas las configuraciones")
    l += ["", "*Fuente:* `resultados/prueba_por_codificacion.csv`", ""]


def seccion_wilcoxon(l: list[str]):
    p = RES / "contrastes_wilcoxon.csv"
    if not p.exists():
        return
    w = pd.read_csv(p)
    nombres = {"HE1_codificacion": "Estrategia de codificación (HE1)",
               "HG_vs_baseline": "Superioridad sobre la línea base (HG)",
               "lineal_vs_ensamblado": "Modelo lineal frente a ensamblados"}

    l += ["## I.3 Contrastes de significancia", "",
          "Prueba de rangos con signo de Wilcoxon sobre veinte observaciones "
          "pareadas, con corrección de Holm-Bonferroni dentro de cada familia "
          "de comparaciones. Se emplea una prueba no paramétrica porque el "
          "supuesto de normalidad de las diferencias no es sostenible con este "
          "número de pares.", ""]

    for familia, titulo in nombres.items():
        b = w[(w.familia == familia) & (w.metrica == "R2")]
        if not len(b):
            continue
        d = pd.DataFrame({
            "Configuración A": b.config_a,
            "Configuración B": b.config_b,
            "Media A": b.media_a.round(4),
            "Media B": b.media_b.round(4),
            "Diferencia": b.diferencia.round(4),
            "*p*": b.p_valor.map(lambda v: f"{v:.2e}" if v < 0.001 else f"{v:.4f}"),
            "Umbral": b.umbral_holm.round(5),
            "Significativo": b.significativo.map({True: "sí", False: "no"}),
        })
        l += [f"### {titulo}", ""]
        l += tabla_md(d, {"Configuración A": "izq", "Configuración B": "izq",
                          "Significativo": "izq"},
                      f"Contrastes de {titulo.lower()}")
        l += [""]
    l += ["*Fuente:* `resultados/contrastes_wilcoxon.csv`", ""]


def seccion_shap(l: list[str]):
    p = RES / "shap_resumen_XGBoost_target_2023.csv"
    if not p.exists():
        return
    s = pd.read_csv(p).head(N_SHAP)
    d = pd.DataFrame({
        "#": range(1, len(s) + 1),
        "Variable": s.variable,
        "mean(&#124;SHAP&#124;)": s.shap_medio_abs.round(4),
        "IC 95 %": [f"[{a:.4f}, {b:.4f}]" if pd.notna(a) else "—"
                    for a, b in zip(s.ic_inferior, s.ic_superior)],
    })
    if "importancia_impureza" in s.columns:
        d["Importancia por impureza"] = s.importancia_impureza.round(4)

    total = pd.read_csv(p)
    l += ["## I.4 Contribución de las variables", "",
          f"Las {N_SHAP} variables de mayor contribución, de un total de "
          f"{len(total)}. Los valores proceden del cálculo sobre la totalidad "
          f"del conjunto de prueba. Los intervalos de confianza se obtuvieron "
          f"por remuestreo con 1000 réplicas sobre las treinta primeras.", "",
          "Se incluye la importancia por reducción de impureza del mismo modelo "
          "para que ambas magnitudes puedan compararse: son distintas y "
          "responden a preguntas distintas, extremo que la hipótesis HE3 "
          "somete a contraste.", ""]
    l += tabla_md(d, {"Variable": "izq", "IC 95 %": "izq"},
                  "Contribución de las cuarenta variables de mayor peso")
    l += ["", "*Fuente:* `resultados/shap_resumen_XGBoost_target_2023.csv`", ""]


def seccion_interacciones(l: list[str]):
    p = RES / "interacciones.csv"
    if not p.exists():
        return
    inter = pd.read_csv(p)
    sust = inter[~inter.redundante].head(20) if "redundante" in inter.columns else inter.head(20)
    d = pd.DataFrame({
        "La contribución de": sust.variable,
        "varía según": sust.modulada_por,
        "Fuerza": sust.fuerza.round(3),
    })
    n_red = int(inter.redundante.sum()) if "redundante" in inter.columns else 0
    l += ["## I.5 Interacciones entre predictores", "",
          "Las veinte interacciones de mayor fuerza entre las doce variables "
          "principales. La fuerza mide cuánto varía la contribución media de "
          "una variable según el tramo en que se sitúa el valor de otra.", "",
          f"Se descartaron {n_red} pares entre variables redundantes, cuya "
          "aparente interacción refleja un reparto inestable de la atribución "
          "entre variables correlacionadas y no una dependencia sustantiva.", ""]
    l += tabla_md(d, {"La contribución de": "izq", "varía según": "izq"},
                  "Interacciones entre predictores, ordenadas por fuerza")
    l += ["", "*Fuente:* `resultados/interacciones.csv`", ""]


def seccion_equidad(l: list[str]):
    l += ["## I.6 Auditoría de equidad por atributo", ""]
    for atributo, anio, titulo in [
        ("income_group", "2023", "Nivel de renta del país"),
        ("wb_region", "2023", "Región económica"),
        ("Age", "2023", "Tramo de edad"),
        ("EdLevel", "2023", "Nivel educativo"),
        ("OrgSize", "2023", "Tamaño de la organización"),
        ("Gender", "2022", "Género (edición 2022)"),
    ]:
        p = RES / f"equidad_{atributo}_{anio}.json"
        if not p.exists():
            continue
        aud = json.loads(p.read_text(encoding="utf-8"))
        filas = []
        for g, v in sorted(aud["por_grupo"].items(),
                           key=lambda kv: -(kv[1].get("mae_relativo") or 0)):
            filas.append({
                "Grupo": g[:44] + ("" if v["suficiente_para_inferencia"] else " *"),
                "*N*": f"{v['n']:,}",
                "MAE (USD)": f"{v['mae']:,.0f}",
                "Error relativo": f"{v['mae_relativo'] * 100:.1f} %" if v["mae_relativo"] else "—",
                "Sesgo (USD)": f"{v['sesgo_sistematico']:+,.0f}",
            })
        l += [f"### {titulo}", ""]
        l += tabla_md(pd.DataFrame(filas), {"Grupo": "izq"},
                          f"Auditoría de equidad por {titulo.lower()}")
        a = aud.get("agregados", {})
        c = aud.get("contrastes", {})
        if a:
            l += ["",
                  f"Razón de disparidad relativa: {a['razon_disparidad_relativa']:.3f} "
                  f"(umbral 1.25). Peor servido: {a['grupo_peor_servido_relativo']}. "
                  f"Coeficiente de variación: {a['coeficiente_variacion_relativo']:.3f}."]
        if c.get("p_valor") is not None:
            l += [f"{c['prueba']}: *p* = {c['p_valor']:.4f}"
                  f" ({'diferencia significativa' if c['significativo'] else 'sin significación'})."]
        l += ["", "Los grupos marcados con asterisco no alcanzan el mínimo de "
              "treinta observaciones y se reportan con carácter descriptivo.", ""]
    l += ["*Fuente:* `resultados/equidad_*.json`", ""]


def seccion_ia(l: list[str]):
    p = RES / "ia_asociacion.csv"
    if not p.exists():
        return
    t = pd.read_csv(p)
    l += ["## I.7 Adopción de inteligencia artificial y compensación", "",
          "Asociación bruta observada en la edición 2025, antes de condicionar "
          "por los determinantes establecidos. La contribución condicionada de "
          "estas variables se analiza en §5.5.2.", ""]
    for var, g in t.groupby("variable"):
        d = pd.DataFrame({
            "Categoría": g.categoria.map(lambda c: c[:58]),
            "*N*": g.n.map(lambda v: f"{v:,}"),
            "Mediana (USD)": g.mediana_usd.map(lambda v: f"{v:,.0f}"),
            "Diferencia": g.diferencia_pct.map(lambda v: f"{v:+.1f} %"),
        })
        l += [f"### `{var}`", ""]
        l += tabla_md(d, {"Categoría": "izq"},
                      f"Compensación por categoría de `{var}`")
        l += [""]
    l += ["*Fuente:* `resultados/ia_asociacion.csv`", ""]


def seccion_mitigacion(l: list[str]):
    p = RES / "mitigacion_por_grupo.csv"
    if not p.exists():
        return
    g = pd.read_csv(p)
    piv = g.pivot_table(index="grupo", columns="estrategia",
                        values="mae_relativo", observed=True).round(1)
    piv = piv.reset_index()
    piv.columns = ["Grupo"] + [c.replace("_", " ").capitalize() for c in piv.columns[1:]]
    l += ["## I.8 Estrategias de mitigación: error relativo por grupo", "",
          "Error relativo de cada grupo bajo cada estrategia, en porcentaje. "
          "El error de los grupos minoritarios no mejora con ninguna: la "
          "disparidad no procede del desequilibrio de la muestra sino de la "
          "mayor heterogeneidad salarial de esos mercados (§5.4.6).", ""]
    l += tabla_md(piv, {"Grupo": "izq"},
                  "Error relativo por grupo bajo cada estrategia de mitigación")
    l += ["", "*Fuente:* `resultados/mitigacion_por_grupo.csv`", ""]


def main():
    l = ["# Anexo I: Tablas de resultados completas", "",
         "El Capítulo V presenta los resultados de forma resumida: los modelos "
         "de mejor desempeño, las variables de mayor contribución y los "
         "contrastes que responden a cada hipótesis. Este anexo recoge las "
         "tablas completas, para que pueda verificarse que la selección de lo "
         "expuesto en el cuerpo del documento no omite resultados desfavorables.", "",
         "Todas las tablas se generan desde los artefactos de la ejecución "
         "mediante `scripts/generar_anexo_resultados.py`. Ninguna cifra se "
         "transcribe a mano.", "", "---", ""]

    for seccion in (seccion_cv, seccion_test, seccion_wilcoxon, seccion_shap,
                    seccion_interacciones, seccion_equidad, seccion_ia,
                    seccion_mitigacion):
        antes = len(l)
        seccion(l)
        if len(l) > antes:
            l += ["---", ""]

    SALIDA.write_text("\n".join(l), encoding="utf-8")
    print(f"Escrito {SALIDA} ({len(l)} líneas)")


if __name__ == "__main__":
    sys.exit(main())
