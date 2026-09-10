#!/usr/bin/env python3
"""Análisis de interacciones mediante valores SHAP.

Ejecuta la segunda parte del criterio de contraste de HE3, que exige documentar
al menos una interacción entre predictores que la importancia por reducción de
impureza no permita identificar.

La medida por impureza asigna a cada variable un único número: cuánto contribuye
a reducir la impureza en el conjunto de particiones del modelo. Por construcción
no puede expresar que el efecto de una variable **dependa del valor de otra**.
Los valores SHAP sí lo permiten, porque asignan una atribución distinta a cada
observación: si esa atribución varía sistemáticamente con el valor de una
segunda variable, hay interacción.

Salida:
    results/interacciones.csv       Fuerza de interacción por par de variables
    figures/shap_dependencia_*.png  Gráficos de dependencia con coloreado

Uso:
    python scripts/run_interacciones.py
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

AZUL, NARANJA, TINTA_2, TINTA_3, REJILLA = "#2a78d6", "#eb6834", "#52514e", "#8a8880", "#e5e4df"


def fuerza_interaccion(valores: np.ndarray, X: np.ndarray, i: int, j: int,
                       n_tramos: int = 5) -> float:
    """Cuantifica cuánto varía la atribución de `i` según el valor de `j`.

    Se divide la muestra en tramos según el valor de la variable `j` y se mide
    la dispersión de la atribución media de `i` entre esos tramos, normalizada
    por su magnitud media. Un valor próximo a cero indica que la contribución de
    `i` no depende de `j`.
    """
    vj = X[:, j]
    if len(np.unique(vj)) < 2:
        return 0.0
    bordes = np.quantile(vj, np.linspace(0, 1, n_tramos + 1))
    bordes = np.unique(bordes)
    if len(bordes) < 3:
        return 0.0
    tramo = np.clip(np.digitize(vj, bordes[1:-1]), 0, len(bordes) - 2)

    medias = [valores[tramo == t, i].mean() for t in range(len(bordes) - 1)
              if (tramo == t).sum() >= 30]
    if len(medias) < 2:
        return 0.0
    escala = np.abs(valores[:, i]).mean()
    return float(np.std(medias) / escala) if escala > 0 else 0.0


def grafico_dependencia(valores, X, nombres, i, j, salida: Path, etiquetas):
    """Gráfico de dependencia de la variable `i`, coloreado por la variable `j`."""
    fig, ax = plt.subplots(figsize=(7, 3.4))
    x, c, y = X[:, i], X[:, j], valores[:, i]

    # Se recorta el color a los percentiles 5 y 95 para que unos pocos valores
    # extremos no comprimen toda la escala cromática.
    lo, hi = np.quantile(c, [0.05, 0.95])
    disp = ax.scatter(x, y, c=np.clip(c, lo, hi), s=7, alpha=0.55,
                      cmap="coolwarm", edgecolors="none")
    barra = fig.colorbar(disp, ax=ax, pad=0.02)
    barra.set_label(etiquetas.get(nombres[j], nombres[j])[:34], fontsize=8, color=TINTA_2)
    barra.ax.tick_params(labelsize=7, colors=TINTA_2)

    ax.axhline(0, color=TINTA_3, linewidth=0.8, zorder=0)
    ax.set_xlabel(etiquetas.get(nombres[i], nombres[i])[:44], fontsize=9, color=TINTA_2)
    ax.set_ylabel(f"Contribución SHAP", fontsize=9, color=TINTA_2)
    ax.set_title(f"Dependencia de {etiquetas.get(nombres[i], nombres[i])[:34]}",
                 loc="left", fontsize=11, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(REJILLA)
    ax.tick_params(labelsize=8, colors=TINTA_2)
    ax.grid(alpha=0.5, color=REJILLA)
    ax.set_axisbelow(True)

    nombre = f"shap_dependencia_{nombres[i].replace('_','').lower()[:20]}"
    ruta = salida / f"{nombre}.png"
    fig.savefig(ruta, dpi=200, bbox_inches="tight", pad_inches=0.15, facecolor="white")
    plt.close(fig)
    return str(ruta)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--etiqueta", default="XGBoost_target_2023")
    ap.add_argument("--top", type=int, default=12,
                    help="Número de variables de mayor contribución a examinar")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--fig-dir", default="figures")
    args = ap.parse_args()

    res = Path(args.results_dir)
    datos = np.load(res / f"shap_values_{args.etiqueta}.npz", allow_pickle=True)
    valores = datos["shap_values"]
    nombres = list(datos["feature_names"])
    resumen = pd.read_csv(res / f"shap_summary_{args.etiqueta}.csv")

    # Se necesita la matriz de predictores transformada para el eje horizontal.
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline

    from src.data_loader_stackoverflow import cargar_encuesta
    from src.feature_pipeline_so import construir_preprocesador, preparar_xy
    from src.model_registry import build_models

    anio = args.etiqueta.split("_")[-1]
    df, _ = cargar_encuesta(anio=anio)
    X, y, _ = preparar_xy(df)
    X_ent, X_pru, y_ent, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=df["income_group"])
    tub = Pipeline([("preprocesador", construir_preprocesador(df, "target")),
                    ("modelo", build_models()["XGBoost"])])
    tub.fit(X_ent, y_ent)
    X_t = np.asarray(tub.named_steps["preprocesador"].transform(X_pru), dtype=float)
    X_muestra = X_t[datos["indices"]]

    principales = [nombres.index(v) for v in resumen.head(args.top)["variable"]
                   if v in nombres]

    etiquetas = {
        "Country": "País (codificado por objetivo)",
        "YearsCodePro_num": "Años de experiencia profesional",
        "YearsCode_num": "Años programando",
        "DevType": "Rol (codificado por objetivo)",
        "WorkExp_num": "Experiencia laboral total",
        "income_group_High income": "País de renta alta",
    }

    print(f"Examinando interacciones entre las {len(principales)} variables principales\n")
    filas = []
    for i in principales:
        for j in principales:
            if i == j:
                continue
            f = fuerza_interaccion(valores, X_muestra, i, j)
            filas.append({"variable": nombres[i], "modulada_por": nombres[j],
                          "fuerza": round(f, 4)})

    inter = pd.DataFrame(filas).sort_values("fuerza", ascending=False)

    # Se marcan los pares redundantes: variables que miden esencialmente lo
    # mismo producen una "interacción" aparente que no es tal, sino un reparto
    # inestable de la atribución entre ellas. Con r = 0.915 entre los años
    # declarados, o entre el país y su nivel de renta —derivado del primero—,
    # la dependencia observada carece de contenido sustantivo.
    REDUNDANTES = [
        {"YearsCode_num", "YearsCodePro_num"},
        {"YearsCode_num", "WorkExp_num"},
        {"YearsCodePro_num", "WorkExp_num"},
        {"Country", "income_group_High income"},
        {"Country", "income_group_Upper middle income"},
        {"Country", "income_group_Lower middle income"},
    ]
    inter["redundante"] = [
        {r.variable, r.modulada_por} in REDUNDANTES for r in inter.itertuples()]
    inter.to_csv(Path(args.results_dir) / "interacciones.csv", index=False)
    sustantivas = inter[~inter.redundante]

    n_red = int(inter.redundante.sum())
    print(f"Descartados {n_red} pares entre variables redundantes "
          f"(colinealidad o derivación directa).\n")
    print("Diez interacciones sustantivas de mayor fuerza:")
    print(f"  {'la contribución de':34s} {'varía según':34s} {'fuerza':>7s}")
    for _, r in sustantivas.head(10).iterrows():
        print(f"  {etiquetas.get(r.variable, r.variable)[:34]:34s} "
              f"{etiquetas.get(r.modulada_por, r.modulada_por)[:34]:34s} {r.fuerza:>7.3f}")

    figs = Path(args.fig_dir)
    figs.mkdir(exist_ok=True)
    print("\nGráficos de dependencia:")
    generados = []
    vistos = set()
    for _, r in sustantivas.iterrows():
        if r.variable in vistos or len(generados) >= 3:
            continue
        i, j = nombres.index(r.variable), nombres.index(r.modulada_por)
        generados.append(grafico_dependencia(valores, X_muestra, nombres, i, j,
                                             figs, etiquetas))
        vistos.add(r.variable)
        print(f"  {generados[-1]}  ({etiquetas.get(r.variable, r.variable)[:28]} "
              f"× {etiquetas.get(r.modulada_por, r.modulada_por)[:28]})")

    print("\nNota: la importancia por reducción de impureza asigna un único valor "
          "por variable\ny no puede expresar ninguna de estas dependencias.")


if __name__ == "__main__":
    main()
