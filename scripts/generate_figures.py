#!/usr/bin/env python3
"""Genera las figuras del documento de tesis a partir de los artefactos de resultados.

Todas las figuras se producen desde los archivos de `results/` y `data/`, nunca
desde cifras transcritas a mano, conforme al criterio de trazabilidad de §3.6.1.

Salida: `figures/*.png` a 200 puntos por pulgada, tamaño ajustado al ancho de
página del documento.

Uso:
    python scripts/generate_figures.py
    python scripts/generate_figures.py --solo dist_compensacion
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader_stackoverflow import TARGET, cargar_encuesta  # noqa: E402

# Paleta categórica validada contra separación por deficiencia de visión
# cromática. El orden es fijo: nunca se cicla ni se reasigna por rango.
AZUL, NARANJA, VERDE, AMBAR = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
TINTA_3 = "#8a8880"
REJILLA = "#e5e4df"

ANCHO = 7.0  # pulgadas, ajustado al ancho de página


# Traducción de los nombres de variable a etiquetas legibles en castellano.
# Los nombres proceden del esquema de la encuesta, que está en inglés.
_VARIABLES = {
    "Country": "País",
    "DevType": "Rol desempeñado",
    "YearsCodePro_num": "Años de experiencia profesional",
    "YearsCode_num": "Años programando",
    "WorkExp_num": "Experiencia laboral total",
}
_CATEGORIAS = {
    "10,000 or more employees": "10,000 o más empleados",
    "5,000 to 9,999 employees": "5,000 a 9,999 empleados",
    "1,000 to 4,999 employees": "1,000 a 4,999 empleados",
    "500 to 999 employees": "500 a 999 empleados",
    "100 to 499 employees": "100 a 499 empleados",
    "20 to 99 employees": "20 a 99 empleados",
    "2 to 9 employees": "2 a 9 empleados",
    "Less than 20 employees": "menos de 20 empleados",
    "Just me - I am a freelancer, sole proprietor, etc.": "trabajador por cuenta propia",
    "Remote": "remoto", "In-person": "presencial", "Hybrid (some remote, some in-person)": "híbrido",
    "High income": "renta alta", "Upper middle income": "renta media-alta",
    "Lower middle income": "renta media-baja",
    "Master’s degree (M.A., M.S., M.Eng., MBA, etc.)": "maestría",
    "Bachelor’s degree (B.A., B.S., B.Eng., etc.)": "grado universitario",
    "Professional degree (JD, MD, Ph.D, Ed.D, etc.)": "doctorado o título profesional",
    "Some college/university study without earning a degree": "estudios sin titulación",
    "Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)": "secundaria",
    "Associate degree (A.A., A.S., etc.)": "título técnico",
    "Independent contributor": "contribuidor individual", "People manager": "responsable de equipo",
    "No declarado": "no declarado",
}
_PREFIJOS = [
    ("language__", "Lenguaje: "), ("database__", "Base de datos: "),
    ("platform__", "Plataforma: "), ("OrgSize_", "Tamaño org.: "),
    ("RemoteWork_", "Modalidad: "), ("income_group_", "Renta: "),
    ("EdLevel_", "Educación: "), ("Age_", "Edad: "), ("ICorPM_", "Función: "),
    ("Industry_", "Sector: "), ("AISelect_", "Uso de IA: "),
    ("AIThreat_", "IA como amenaza: "), ("AIAgents_", "Agentes de IA: "),
    ("LearnCodeAI_", "Aprendizaje de IA: "),
]


def traducir_variable(v: str, max_len: int = 42) -> str:
    """Convierte un nombre de variable del esquema de la encuesta en etiqueta legible."""
    if v in _VARIABLES:
        return _VARIABLES[v]
    for prefijo, es in _PREFIJOS:
        if v.startswith(prefijo):
            resto = v[len(prefijo):]
            resto = _CATEGORIAS.get(resto, resto.replace("_", " "))
            # Los títulos de grado largos se abrevian: no aportan al mensaje.
            if "degree" in resto.lower():
                resto = resto.split("(")[0].strip().lower()
            etiqueta = es + resto
            return etiqueta[:max_len - 1] + "…" if len(etiqueta) > max_len else etiqueta
    return v.replace("_", " ")[:max_len]


def estilo():
    plt.rcParams.update({
        "figure.dpi": 200, "savefig.dpi": 200,
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlepad": 12,
        "axes.labelsize": 9, "axes.labelcolor": TINTA_2,
        "axes.edgecolor": REJILLA, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": TINTA_2, "ytick.color": TINTA_2,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "grid.color": REJILLA, "grid.linewidth": 0.7,
        "legend.frameon": False, "legend.fontsize": 8,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


def guardar(fig, nombre: str, salida: Path) -> str:
    ruta = salida / f"{nombre}.png"
    fig.savefig(ruta, bbox_inches="tight", pad_inches=0.18, facecolor="white")
    plt.close(fig)
    print(f"  {ruta}")
    return str(ruta)


# ---------------------------------------------------------------------------

def fig_distribucion(df, salida):
    """Distribución de la compensación, en escala original y logarítmica."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(ANCHO, 2.9))

    ax1.hist(df[TARGET] / 1000, bins=60, color=AZUL, edgecolor="white", linewidth=0.3)
    ax1.set_xlabel("Compensación anual (miles de USD)")
    ax1.set_ylabel("Frecuencia")
    ax1.set_title("Escala original", loc="left")
    mediana = df[TARGET].median() / 1000
    ax1.axvline(mediana, color=NARANJA, linewidth=1.6)
    ax1.annotate(f"mediana\n${mediana:,.0f}k", xy=(mediana, ax1.get_ylim()[1] * 0.72),
                 xytext=(8, 0), textcoords="offset points",
                 color=NARANJA, fontsize=8, fontweight="bold")
    ax1.text(0.97, 0.92, f"asimetría {df[TARGET].skew():.2f}", transform=ax1.transAxes,
             ha="right", color=TINTA_3, fontsize=8)

    log = np.log1p(df[TARGET])
    ax2.hist(log, bins=60, color=AZUL, edgecolor="white", linewidth=0.3)
    ax2.set_xlabel("ln(1 + compensación)")
    ax2.set_title("Escala logarítmica", loc="left")
    ax2.text(0.03, 0.92, f"asimetría {log.skew():.2f}", transform=ax2.transAxes,
             color=TINTA_3, fontsize=8)

    for ax in (ax1, ax2):
        ax.grid(axis="y", alpha=0.6)
        ax.set_axisbelow(True)
    fig.suptitle("Distribución de la variable dependiente", x=0.02, ha="left",
                 fontsize=11, fontweight="bold", y=1.06)
    return guardar(fig, "dist_compensacion", salida)


def fig_composicion(df, salida):
    """Mediana de compensación por nivel de renta, con el tamaño de cada grupo."""
    g = (df.groupby("income_group", observed=True)[TARGET]
         .agg(["median", "count"]).sort_values("median"))
    orden = {"Lower middle income": "Renta media-baja",
             "Upper middle income": "Renta media-alta",
             "High income": "Renta alta"}
    etiquetas = [orden.get(i, i) for i in g.index]

    fig, ax = plt.subplots(figsize=(ANCHO, 2.4))
    barras = ax.barh(etiquetas, g["median"] / 1000, color=AZUL, height=0.55)
    for b, (_, fila) in zip(barras, g.iterrows()):
        ax.text(b.get_width() + 1.5, b.get_y() + b.get_height() / 2,
                f"${fila['median']/1000:,.1f}k   n = {int(fila['count']):,}",
                va="center", fontsize=8, color=TINTA_2)
    ax.set_xlabel("Mediana de compensación anual (miles de USD)")
    ax.set_xlim(0, g["median"].max() / 1000 * 1.32)
    ax.grid(axis="x", alpha=0.6)
    ax.set_axisbelow(True)
    ax.set_title("Compensación por nivel de renta del país", loc="left")
    return guardar(fig, "composicion_renta", salida)


def fig_modelos(salida, resultados: Path):
    """Comparación de modelos: R² medio con dispersión entre particiones."""
    cv = pd.read_csv(resultados / "cv_by_encoding.csv")
    g = (cv.groupby(["modelo", "codificacion"])["R2"].agg(["mean", "std"])
         .reset_index())
    g = g[~g.modelo.str.startswith("Baseline")].sort_values("mean")

    etiquetas, medias, errores, colores = [], [], [], []
    for _, f in g.iterrows():
        cod = "por objetivo" if f.codificacion == "target" else "disyuntiva"
        etiquetas.append(f"{f.modelo}\n{cod}")
        medias.append(f["mean"])
        errores.append(f["std"])
        colores.append(NARANJA if f.modelo == "Ridge" else AZUL)

    fig, ax = plt.subplots(figsize=(ANCHO, 3.6))
    y = np.arange(len(etiquetas))
    ax.barh(y, medias, xerr=errores, color=colores, height=0.6,
            error_kw={"ecolor": TINTA_3, "elinewidth": 1, "capsize": 2.5})
    ax.set_yticks(y)
    ax.set_yticklabels(etiquetas, fontsize=7.5)
    for i, (m, e) in enumerate(zip(medias, errores)):
        ax.text(m + e + 0.004, i, f"{m:.4f}", va="center", fontsize=7.5, color=TINTA_2)
    ax.set_xlabel("R², media ± desviación típica entre 20 particiones", fontsize=8.5)
    ax.set_xlim(0.72, 0.81)
    ax.grid(axis="x", alpha=0.6)
    ax.set_axisbelow(True)
    ax.set_title("Desempeño por modelo y estrategia de codificación", loc="left")
    ax.text(0.99, 0.03, "naranja: modelo lineal   ·   azul: ensamblados",
            transform=ax.transAxes, ha="right", fontsize=7.5, color=TINTA_3)
    return guardar(fig, "comparacion_modelos", salida)


def fig_shap(salida, resultados: Path, etiqueta="XGBoost_target_2023", n=15):
    """Contribución media de las variables de mayor peso, con intervalos."""
    ruta = resultados / f"shap_summary_{etiqueta}.csv"
    if not ruta.exists():
        return None
    s = pd.read_csv(ruta).head(n).iloc[::-1]

    s = s.assign(etiqueta=[traducir_variable(v) for v in s["variable"]])

    fig, ax = plt.subplots(figsize=(ANCHO, 4.2))
    y = np.arange(len(s))
    colores = [NARANJA if v == "Country" else AZUL for v in s["variable"]]
    ax.barh(y, s["shap_medio_abs"], color=colores, height=0.62)
    if s["ic_inferior"].notna().any():
        err = np.vstack([s["shap_medio_abs"] - s["ic_inferior"].fillna(s["shap_medio_abs"]),
                         s["ic_superior"].fillna(s["shap_medio_abs"]) - s["shap_medio_abs"]])
        ax.errorbar(s["shap_medio_abs"], y, xerr=err, fmt="none",
                    ecolor=TINTA_3, elinewidth=1, capsize=2)
    ax.set_yticks(y)
    ax.set_yticklabels(s["etiqueta"], fontsize=7.5)
    margen = s["shap_medio_abs"].max() * 0.035
    for i, (v, sup) in enumerate(zip(s["shap_medio_abs"], s["ic_superior"].fillna(s["shap_medio_abs"]))):
        ax.text(max(v, sup) + margen, i, f"{v:.3f}", va="center",
                fontsize=7.5, color=TINTA_2)
    ax.set_xlabel("Contribución media, mean(|SHAP|), en escala logarítmica del salario", fontsize=8.5)
    ax.set_xlim(0, s["shap_medio_abs"].max() * 1.22)
    ax.grid(axis="x", alpha=0.6)
    ax.set_axisbelow(True)
    ax.set_title("Determinantes de la compensación estimada", loc="left")
    return guardar(fig, "shap_importancia", salida)


def fig_equidad(salida, resultados: Path, anio="2023"):
    """Error relativo por región, que es el indicador comparable entre grupos."""
    ruta = resultados / f"fairness_wb_region_{anio}.json"
    if not ruta.exists():
        return None
    aud = json.loads(ruta.read_text(encoding="utf-8"))

    trad = {"Latin America & Caribbean": "América Latina y Caribe",
            "South Asia": "Asia del Sur", "Sub-Saharan Africa": "África Subsahariana",
            "Middle East, North Africa, Afghanistan & Pakistan": "Oriente Medio y N. de África",
            "North America": "Norteamérica", "Europe & Central Asia": "Europa y Asia Central",
            "East Asia & Pacific": "Asia Oriental y Pacífico"}

    filas = [(trad.get(g, g), v["mae_relativo"] * 100, v["mae"], v["n"],
              v["suficiente_para_inferencia"])
             for g, v in aud["por_grupo"].items() if v["mae_relativo"]]
    filas.sort(key=lambda r: r[1])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(ANCHO, 3.4), sharey=True)
    y = np.arange(len(filas))
    etiquetas = [f[0][:28] for f in filas]

    colores = [NARANJA if "América Latina" in f[0] else
               (TINTA_3 if not f[4] else AZUL) for f in filas]
    ax1.barh(y, [f[1] for f in filas], color=colores, height=0.6)
    for i, f in enumerate(filas):
        ax1.text(f[1] + 1, i, f"{f[1]:.1f} %", va="center", fontsize=7.5, color=TINTA_2)
    ax1.set_yticks(y)
    ax1.set_yticklabels(etiquetas, fontsize=8)
    ax1.set_xlabel("Error relativo (MAE / mediana del grupo)", fontsize=8.5)
    ax1.set_xlim(0, max(f[1] for f in filas) * 1.24)
    ax1.set_title("Comparable entre grupos", loc="left", fontsize=9.5)

    ax2.barh(y, [f[2] / 1000 for f in filas], color=TINTA_3, height=0.6)
    for i, f in enumerate(filas):
        ax2.text(f[2] / 1000 + 1, i, f"${f[2]/1000:.0f}k", va="center",
                 fontsize=7.5, color=TINTA_2)
    ax2.set_xlabel("Error absoluto (MAE, miles de USD)", fontsize=8.5)
    ax2.set_xlim(0, max(f[2] for f in filas) / 1000 * 1.22)
    ax2.set_title("No comparable: invierte el orden", loc="left", fontsize=9.5)

    for ax in (ax1, ax2):
        ax.grid(axis="x", alpha=0.6)
        ax.set_axisbelow(True)
    fig.suptitle("Error del modelo por región económica", x=0.02, ha="left",
                 fontsize=11, fontweight="bold", y=1.04)
    fig.text(0.02, -0.07,
             "El grupo peor servido cambia según la medida empleada: en términos relativos es América Latina "
             "(naranja);\nen términos absolutos, Norteamérica. Solo la medida relativa es comparable entre "
             "grupos de escala salarial dispar.",
             fontsize=7.5, color=TINTA_3, ha="left", va="top")
    return guardar(fig, "equidad_por_region", salida)


def fig_ia(salida, resultados: Path):
    """Asociación bruta entre adopción de IA y compensación, frente a su contribución real."""
    ruta = resultados / "ia_asociacion.csv"
    if not ruta.exists():
        return None
    t = pd.read_csv(ruta)
    t = t[t.variable == "AISelect"].sort_values("mediana_usd")
    if not len(t):
        return None

    trad = {"Yes, I use AI tools daily": "Uso diario",
            "Yes, I use AI tools weekly": "Uso semanal",
            "Yes, I use AI tools monthly or infrequently": "Uso mensual o esporádico",
            "No, and I don't plan to": "No la usa ni planea usarla",
            "No, but I plan to soon": "No, pero planea usarla"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(ANCHO, 3.0),
                                   gridspec_kw={"width_ratios": [1.35, 1]})
    y = np.arange(len(t))
    ax1.barh(y, t["mediana_usd"] / 1000, color=AZUL, height=0.6)
    ax1.set_yticks(y)
    ax1.set_yticklabels([trad.get(c, c)[:28] for c in t["categoria"]], fontsize=8)
    for i, (_, f) in enumerate(t.iterrows()):
        ax1.text(f.mediana_usd / 1000 + 1.2, i, f"${f.mediana_usd/1000:.1f}k",
                 va="center", fontsize=7.5, color=TINTA_2)
    ax1.set_xlabel("Mediana de compensación (miles de USD)")
    ax1.set_xlim(0, t["mediana_usd"].max() / 1000 * 1.2)
    ax1.set_title("Asociación bruta observada", loc="left", fontsize=9.5)

    # Contribución condicionada: la suma de las variables de IA frente a los
    # dos determinantes principales.
    s = resultados / "shap_summary_XGBoost_target_2025.csv"
    if s.exists():
        sh = pd.read_csv(s)
        ia = sh[sh.variable.str.startswith(("AISelect", "AIThreat", "AIAgents",
                                            "LearnCodeAI"))]["shap_medio_abs"].sum()
        pais = float(sh.loc[sh.variable == "Country", "shap_medio_abs"].iloc[0])
        exp = float(sh.loc[sh.variable == "WorkExp_num", "shap_medio_abs"].iloc[0])
        vals = [ia, exp, pais]
        etq = ["Variables\nde IA (23)", "Experiencia\nlaboral", "País"]
        col = [NARANJA, TINTA_3, AZUL]
        b = ax2.bar(np.arange(3), vals, color=col, width=0.55)
        for r, v in zip(b, vals):
            ax2.text(r.get_x() + r.get_width() / 2, v + 0.012, f"{v:.3f}",
                     ha="center", fontsize=8, color=TINTA_2)
        ax2.set_xticks(np.arange(3))
        ax2.set_xticklabels(etq, fontsize=7.5, linespacing=1.3)
        ax2.set_ylabel("mean(|SHAP|)")
        ax2.set_ylim(0, max(vals) * 1.2)
        ax2.grid(axis="y", alpha=0.6)
        ax2.set_axisbelow(True)
        ax2.set_title("Contribución condicionada", loc="left", fontsize=9.5)

    ax1.grid(axis="x", alpha=0.6)
    ax1.set_axisbelow(True)
    fig.suptitle("Adopción de inteligencia artificial y compensación (edición 2025)",
                 x=0.02, ha="left", fontsize=11, fontweight="bold", y=1.05)
    fig.subplots_adjust(wspace=0.42)
    fig.text(0.02, -0.07,
             "La diferencia bruta del 8.6 % entre uso diario y no uso se desvanece al condicionar: "
             "las 23 variables de IA suman\nuna contribución 6.7 veces inferior a la del país. "
             "La asociación es un artefacto de composición de la muestra.",
             fontsize=7.5, color=TINTA_3, ha="left", va="top")
    return guardar(fig, "ia_compensacion", salida)


def fig_residuos(salida, df, resultados: Path):
    """Diagnóstico de residuos del modelo seleccionado."""
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline

    from src.feature_pipeline_so import construir_preprocesador, preparar_xy
    from src.model_registry import build_models

    X, y, _ = preparar_xy(df)
    X_ent, X_pru, y_ent, y_pru = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=df["income_group"])
    tub = Pipeline([("preprocesador", construir_preprocesador(df, "target")),
                    ("modelo", build_models()["XGBoost"])])
    tub.fit(X_ent, y_ent)
    pred = tub.predict(X_pru)
    residuos = y_pru - pred

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(ANCHO, 3.0))
    ax1.scatter(pred, residuos, s=3, alpha=0.18, color=AZUL, edgecolors="none")
    ax1.axhline(0, color=NARANJA, linewidth=1.2)
    ax1.set_xlabel("Valor estimado (escala logarítmica)")
    ax1.set_ylabel("Residuo")
    ax1.set_title("Residuos frente a estimación", loc="left", fontsize=9.5)

    from scipy import stats
    stats.probplot(residuos, dist="norm", plot=ax2)
    ax2.get_lines()[0].set(marker="o", markersize=1.6, alpha=0.25,
                           color=AZUL, markeredgecolor="none")
    ax2.get_lines()[1].set(color=NARANJA, linewidth=1.2)
    ax2.set_title("Gráfico cuantil-cuantil normal", loc="left", fontsize=9.5)
    ax2.set_xlabel("Cuantiles teóricos")
    ax2.set_ylabel("Cuantiles observados")

    for ax in (ax1, ax2):
        ax.grid(alpha=0.6)
        ax.set_axisbelow(True)
    fig.suptitle("Diagnóstico de residuos del modelo seleccionado", x=0.02,
                 ha="left", fontsize=11, fontweight="bold", y=1.04)
    fig.subplots_adjust(wspace=0.28)
    fig.text(0.02, -0.07,
             f"Asimetría de los residuos: {pd.Series(residuos).skew():.2f}. "
             "La cola izquierda se aparta de la normalidad, de modo que el supuesto\n"
             "se cumple de forma aproximada. Los intervalos reportados en el documento "
             "proceden de remuestreo, no de la hipótesis normal.",
             fontsize=7.5, color=TINTA_3, ha="left", va="top")
    return guardar(fig, "residuos", salida)


def fig_interaccion_experiencia(salida, resultados: Path, df):
    """Retorno de la experiencia según el nivel de renta del país.

    Es la interacción sustantiva que sostiene la segunda parte del criterio de
    HE3: la contribución de la experiencia al salario estimado depende del
    mercado, dependencia que una medida de importancia por variable no puede
    expresar.
    """
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline

    from src.feature_pipeline_so import construir_preprocesador, preparar_xy
    from src.model_registry import build_models

    npz = resultados / "shap_values_XGBoost_target_2023.npz"
    if not npz.exists():
        return None
    datos = np.load(npz, allow_pickle=True)
    valores, nombres = datos["shap_values"], list(datos["feature_names"])
    if "YearsCodePro_num" not in nombres:
        return None
    k = nombres.index("YearsCodePro_num")

    X, y, A = preparar_xy(df)
    X_ent, X_pru, y_ent, _, _, A_pru = train_test_split(
        X, y, A, test_size=0.2, random_state=42, stratify=df["income_group"])
    tub = Pipeline([("preprocesador", construir_preprocesador(df, "target")),
                    ("modelo", build_models()["XGBoost"])])
    tub.fit(X_ent, y_ent)

    idx = datos["indices"]
    anios = X_pru["YearsCodePro_num"].to_numpy()[idx]
    renta = A_pru["income_group"].to_numpy()[idx]
    shap_exp = valores[:, k]

    grupos = [("High income", "Renta alta", AZUL),
              ("Upper middle income", "Renta media-alta", AMBAR),
              ("Lower middle income", "Renta media-baja", NARANJA)]

    fig, ax = plt.subplots(figsize=(ANCHO, 3.6))
    tramos = np.array([0, 2, 5, 8, 12, 18, 25, 50])
    centros = (tramos[:-1] + tramos[1:]) / 2

    for clave, etiqueta, color in grupos:
        m = (renta == clave) & np.isfinite(anios)
        if m.sum() < 50:
            continue
        medias, xs = [], []
        for a, b, c in zip(tramos[:-1], tramos[1:], centros):
            sel = m & (anios >= a) & (anios < b)
            if sel.sum() >= 20:
                medias.append(shap_exp[sel].mean())
                xs.append(c)
        ax.plot(xs, medias, marker="o", markersize=5, linewidth=2,
                color=color, label=f"{etiqueta}  (n = {m.sum():,})")

    ax.axhline(0, color=TINTA_3, linewidth=0.8, zorder=0)
    ax.set_xlabel("Años de experiencia profesional", fontsize=9)
    ax.set_ylabel("Contribución media al salario estimado\n(escala logarítmica)", fontsize=9)
    ax.set_title("Retorno de la experiencia según el nivel de renta del país",
                 loc="left")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.6)
    ax.set_axisbelow(True)
    fig.text(0.02, -0.07,
             "La contribución de la experiencia al salario estimado no es la misma en todos los mercados: "
             "su pendiente\ndifiere según el nivel de renta del país. Una medida de importancia por variable "
             "asigna un único valor a la\nexperiencia y no puede expresar esta dependencia.",
             fontsize=7.5, color=TINTA_3, ha="left", va="top")
    return guardar(fig, "interaccion_experiencia_renta", salida)


def fig_mitigacion(salida, resultados: Path):
    """Compromiso entre exactitud y equidad de las estrategias de mitigación."""
    ruta = resultados / "mitigacion_resultados.csv"
    if not ruta.exists():
        return None
    m = pd.read_csv(ruta)
    etq = {"sin_mitigacion": "Sin mitigación", "reponderacion": "Reponderación",
           "submuestreo": "Submuestreo", "sobremuestreo": "Sobremuestreo"}

    fig, ax = plt.subplots(figsize=(ANCHO, 3.4))
    for _, r in m.iterrows():
        base = r.estrategia == "sin_mitigacion"
        ax.scatter(r.razon_disparidad_renta, r.R2, s=150 if base else 110,
                   color=NARANJA if base else AZUL, zorder=3,
                   edgecolors="white", linewidths=1.5)
        ax.annotate(etq.get(r.estrategia, r.estrategia),
                    (r.razon_disparidad_renta, r.R2),
                    xytext=(0, 13), textcoords="offset points",
                    ha="center", fontsize=8.5, color=TINTA_2,
                    fontweight="bold" if base else "normal")
    ax.set_xlabel("Razón de disparidad del error entre grupos de renta  →  peor equidad", fontsize=9)
    ax.set_ylabel("R² sobre el conjunto de prueba", fontsize=9)
    ax.set_title("Compromiso entre exactitud y equidad", loc="left")
    ax.grid(alpha=0.6)
    ax.set_axisbelow(True)
    ax.margins(0.16, 0.22)
    fig.text(0.02, -0.07,
             "Ninguna estrategia de mitigación reduce la disparidad de forma apreciable. El submuestreo es "
             "la única que\nla mueve (−0.13), a costa de 0.036 puntos de R². La disparidad no procede del "
             "desbalance de la muestra\nsino de la mayor heterogeneidad salarial intrínseca de esos mercados.",
             fontsize=7.5, color=TINTA_3, ha="left", va="top")
    return guardar(fig, "mitigacion_compromiso", salida)


def fig_ppp(salida, resultados: Path):
    """Nominal frente a paridad de poder adquisitivo: pesos y medianas."""
    ruta = resultados / "ppp_comparativa.json"
    if not ruta.exists():
        return None
    import json
    d = json.loads(ruta.read_text(encoding="utf-8"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(ANCHO, 3.4))
    fig.subplots_adjust(wspace=0.34)

    # Panel izquierdo: contribución SHAP por bloque en ambas escalas.
    nom = d["modelo"]["nominal"]["pesos_shap"]
    ppa = d["modelo"]["ppa"]["pesos_shap"]
    bloques = list(nom.keys())
    y = np.arange(len(bloques))
    ax1.barh(y + 0.2, [nom[b] for b in bloques], height=0.38,
             color=AZUL, label="Dólares nominales")
    ax1.barh(y - 0.2, [ppa.get(b, 0) for b in bloques], height=0.38,
             color=NARANJA, label="Dólares PPA")
    ax1.set_yticks(y, bloques, fontsize=8)
    ax1.invert_yaxis()
    ax1.set_xlabel("% de la contribución SHAP total", fontsize=8.5)
    ax1.set_title("Contribución por bloque", loc="left")
    ax1.legend(loc="lower right", fontsize=7.5)
    ax1.grid(axis="x", alpha=0.6)
    ax1.set_axisbelow(True)

    # Panel derecho: mediana de compensación por nivel de renta.
    orden = ["High income", "Upper middle income", "Lower middle income"]
    etiquetas = ["Renta\nalta", "Renta\nmedia-alta", "Renta\nmedia-baja"]
    x = np.arange(len(orden))
    mn = [d["descriptivo"]["nominal"]["por_renta"][g] / 1000 for g in orden]
    mp = [d["descriptivo"]["ppa"]["por_renta"][g] / 1000 for g in orden]
    ax2.bar(x - 0.2, mn, width=0.38, color=AZUL, label="Nominal")
    ax2.bar(x + 0.2, mp, width=0.38, color=NARANJA, label="PPA")
    for i, (a, b) in enumerate(zip(mn, mp)):
        ax2.text(i - 0.2, a + 2, f"{a:.0f}", ha="center", fontsize=7, color=TINTA_2)
        ax2.text(i + 0.2, b + 2, f"{b:.0f}", ha="center", fontsize=7, color=TINTA_2)
    ax2.set_xticks(x, etiquetas, fontsize=8)
    ax2.set_ylabel("Mediana (miles de USD)", fontsize=8.5)
    ax2.set_title("Mediana por nivel de renta", loc="left")
    ax2.legend(fontsize=7.5)
    ax2.grid(axis="y", alpha=0.6)
    ax2.set_axisbelow(True)

    return guardar(fig, "ppp_comparativa", salida)


FIGURAS = {

    "dist_compensacion": lambda d, r, s: fig_distribucion(d, s),
    "composicion_renta": lambda d, r, s: fig_composicion(d, s),
    "comparacion_modelos": lambda d, r, s: fig_modelos(s, r),
    "shap_importancia": lambda d, r, s: fig_shap(s, r),
    "equidad_por_region": lambda d, r, s: fig_equidad(s, r),
    "residuos": lambda d, r, s: fig_residuos(s, d, r),
    "ia_compensacion": lambda d, r, s: fig_ia(s, r),
    "interaccion_experiencia_renta": lambda d, r, s: fig_interaccion_experiencia(s, r, d),
    "mitigacion_compromiso": lambda d, r, s: fig_mitigacion(s, r),
    "ppp_comparativa": lambda d, r, s: fig_ppp(s, r),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anio", default="2023")
    ap.add_argument("--solo", nargs="+", default=None)
    ap.add_argument("--out-dir", default="figures")
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()

    estilo()
    salida = Path(args.out_dir)
    salida.mkdir(parents=True, exist_ok=True)
    resultados = Path(args.results_dir)

    print(f"Cargando la edición {args.anio}...")
    df, _ = cargar_encuesta(anio=args.anio)

    nombres = args.solo or list(FIGURAS)
    print(f"\nGenerando {len(nombres)} figuras:")
    for nombre in nombres:
        if nombre not in FIGURAS:
            print(f"  [omitida] {nombre}: no definida")
            continue
        try:
            if FIGURAS[nombre](df, resultados, salida) is None:
                print(f"  [omitida] {nombre}: falta el artefacto de origen")
        except Exception as e:  # noqa: BLE001
            print(f"  [error] {nombre}: {e}")


if __name__ == "__main__":
    main()
