"""Valores SHAP del modelo ajustado.

Usa el estimador específico de cada familia (TreeExplainer, LinearExplainer);
el de núcleo queda como último recurso porque no escala a decenas de miles de
filas. Exporta los valores crudos, no solo los gráficos, y también la
importancia por impureza para poder comparar las dos.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _elegir_estimador(modelo, X_fondo: np.ndarray):
    """Estimador SHAP según la familia del modelo."""
    import shap

    familia_arbol = (
        "HistGradientBoostingRegressor", "XGBRegressor", "LGBMRegressor",
        "CatBoostRegressor", "RandomForestRegressor", "ExtraTreesRegressor",
        "GradientBoostingRegressor", "DecisionTreeRegressor",
    )
    familia_lineal = ("Ridge", "Lasso", "ElasticNet", "LinearRegression")
    nombre = type(modelo).__name__

    if nombre in familia_arbol:
        return shap.TreeExplainer(modelo), "TreeExplainer"
    if nombre in familia_lineal:
        return shap.LinearExplainer(modelo, X_fondo), "LinearExplainer"

    print(f"  Aviso: no hay estimador específico para {nombre}; "
          f"se recurre al basado en núcleo, de coste considerable.")
    return shap.KernelExplainer(modelo.predict, shap.sample(X_fondo, 100)), "KernelExplainer"


def importancia_por_impureza(modelo, nombres: list[str]) -> dict | None:
    """Importancia por impureza (o |coef| en los lineales), si el modelo la expone.

    No es lo mismo que SHAP: mide cuánto usa el modelo la variable para
    partir, no cuánto aporta a cada predicción.
    """
    if hasattr(modelo, "feature_importances_"):
        valores = np.asarray(modelo.feature_importances_, dtype=float)
    elif hasattr(modelo, "coef_"):
        valores = np.abs(np.asarray(modelo.coef_, dtype=float)).ravel()
    else:
        return None
    if len(valores) != len(nombres):
        return None
    return {n: float(v) for n, v in zip(nombres, valores)}


def explicar(tuberia, X_entrenamiento: pd.DataFrame, X_prueba: pd.DataFrame,
             nombres: list[str] | None = None,
             directorio: str = "resultados", etiqueta: str = "modelo",
             n_fondo: int = 500, n_explicar: int = 20000,
             semilla: int = 42) -> dict:
    """Calcula y exporta los valores SHAP de una tubería ya ajustada.

    Devuelve la importancia media por variable y las rutas de los artefactos.
    """
    import shap

    salida = Path(directorio)
    salida.mkdir(parents=True, exist_ok=True)

    preprocesador = tuberia.named_steps["preprocesador"]
    modelo = tuberia.named_steps["modelo"]

    X_ent_t = np.asarray(preprocesador.transform(X_entrenamiento), dtype=float)
    X_pru_t = np.asarray(preprocesador.transform(X_prueba), dtype=float)

    if nombres is None or len(nombres) != X_ent_t.shape[1]:
        nombres = [f"x{i}" for i in range(X_ent_t.shape[1])]

    rng = np.random.default_rng(semilla)
    if len(X_ent_t) > n_fondo:
        X_fondo = X_ent_t[rng.choice(len(X_ent_t), n_fondo, replace=False)]
    else:
        X_fondo = X_ent_t

    # Muestra con semilla fija y tamaño registrado. El valor por defecto cubre
    # todo el conjunto de prueba; con 2,000 los estratos de renta baja se
    # quedaban en ~100 casos.
    if len(X_pru_t) > n_explicar:
        idx = rng.choice(len(X_pru_t), n_explicar, replace=False)
        X_muestra = X_pru_t[idx]
    else:
        idx = np.arange(len(X_pru_t))
        X_muestra = X_pru_t

    estimador, tipo = _elegir_estimador(modelo, X_fondo)
    print(f"  [{etiqueta}] {tipo} sobre {len(X_muestra):,} observaciones")

    valores = estimador.shap_values(X_muestra)
    if isinstance(valores, list):
        valores = valores[0]
    valores = np.asarray(valores)

    media_abs = np.abs(valores).mean(axis=0)
    orden = np.argsort(media_abs)[::-1]

    # IC por bootstrap de la importancia media, solo para las 30 primeras.
    n_replicas = 1000
    ic = {}
    for i in orden[:30]:
        muestras = np.abs(valores[:, i])
        replicas = np.array([
            muestras[rng.integers(0, len(muestras), len(muestras))].mean()
            for _ in range(n_replicas)
        ])
        ic[nombres[i]] = (float(np.percentile(replicas, 2.5)),
                          float(np.percentile(replicas, 97.5)))

    ruta_valores = salida / f"shap_values_{etiqueta}.npz"
    np.savez_compressed(ruta_valores, shap_values=valores,
                        indices=idx, feature_names=np.array(nombres, dtype=object))

    resumen = pd.DataFrame({
        "variable": [nombres[i] for i in orden],
        "shap_medio_abs": [float(media_abs[i]) for i in orden],
    })
    resumen["ic_inferior"] = resumen["variable"].map(lambda v: ic.get(v, (None, None))[0])
    resumen["ic_superior"] = resumen["variable"].map(lambda v: ic.get(v, (None, None))[1])

    impureza = importancia_por_impureza(modelo, nombres)
    if impureza:
        resumen["importancia_impureza"] = resumen["variable"].map(impureza)

    ruta_resumen = salida / f"shap_resumen_{etiqueta}.csv"
    resumen.to_csv(ruta_resumen, index=False)

    salida_dict = {
        "estimador": tipo,
        "n_explicadas": int(len(X_muestra)),
        "n_variables": int(len(nombres)),
        "top_10": resumen.head(10).to_dict("records"),
        "ruta_valores": str(ruta_valores),
        "ruta_resumen": str(ruta_resumen),
    }

    # Correlación de rangos entre los dos ordenamientos.
    if impureza:
        from scipy.stats import spearmanr
        comunes = resumen.dropna(subset=["importancia_impureza"])
        if len(comunes) > 2:
            rho, p = spearmanr(comunes["shap_medio_abs"], comunes["importancia_impureza"])
            salida_dict["spearman_shap_vs_impureza"] = {
                "rho": round(float(rho), 4), "p_valor": round(float(p), 6),
                "n_variables": int(len(comunes)),
            }

    return salida_dict


def grafico_resumen(ruta_npz: str, ruta_salida: str, max_variables: int = 20) -> str:
    """Gráfico resumen de SHAP a partir del .npz exportado."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    datos = np.load(ruta_npz, allow_pickle=True)
    valores = datos["shap_values"]
    nombres = list(datos["feature_names"])

    plt.figure(figsize=(10, 7))
    shap.summary_plot(valores, features=None, feature_names=nombres,
                      show=False, max_display=max_variables)
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150, bbox_inches="tight")
    plt.close()
    return ruta_salida
