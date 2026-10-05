"""Preprocesador de la encuesta: numéricas, categóricas y tecnologías.

Ofrece las dos representaciones de las categóricas de alta cardinalidad que
se comparan en el experimento: one-hot y codificación por objetivo. Todo lo
que estima parámetros (imputación, escalado, codificación por objetivo) va
dentro de la tubería para que la validación cruzada lo reajuste en cada
partición; si no, hay fuga de información.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.codificadores import CodificadorPorObjetivo

# Alta cardinalidad: las que se someten a las dos codificaciones.
CAT_ALTA = ["Country", "DevType"]

# Pocas categorías: siempre one-hot.
CAT_BAJA = ["EdLevel", "OrgSize", "RemoteWork", "Industry", "ICorPM",
            "Age", "income_group"]

# Preguntas sobre inteligencia artificial: AISelect desde 2023, las cuatro en 2025.
CAT_IA = ["AISelect", "AIThreat", "AIAgents", "LearnCodeAI"]

NUMERICAS = ["YearsCode_num", "YearsCodePro_num", "WorkExp_num"]

# Indicadores binarios que genera carga_encuesta.expandir_multivalor.
PREFIJOS_TECNOLOGIA = ("language__", "database__", "platform__")

OBJETIVO = "salary_log"


def columnas_tecnologia(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(PREFIJOS_TECNOLOGIA)]


def columnas_disponibles(df: pd.DataFrame, candidatas: list[str]) -> list[str]:
    """Solo las columnas presentes; el esquema cambia entre ediciones."""
    return [c for c in candidatas if c in df.columns]


def construir_preprocesador(df: pd.DataFrame, codificacion: str = "target",
                            suavizado: float = 10.0) -> ColumnTransformer:
    """Transformador de columnas; `codificacion` es 'target' u 'onehot' para CAT_ALTA."""
    num = columnas_disponibles(df, NUMERICAS)
    baja = columnas_disponibles(df, CAT_BAJA + CAT_IA)
    alta = columnas_disponibles(df, CAT_ALTA)
    tec = columnas_tecnologia(df)

    t_num = Pipeline([
        ("imputador", SimpleImputer(strategy="median")),
        ("escalado", StandardScaler()),
    ])

    # Sin imputar la moda: la carga ya etiqueta el ausente como categoría
    # propia, y la ausencia puede ser informativa.
    t_baja = OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                           min_frequency=30)

    if codificacion == "onehot":
        t_alta = OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                               min_frequency=30)
    elif codificacion == "target":
        t_alta = Pipeline([
            # El codificador espera códigos numéricos, no texto.
            ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value",
                                       unknown_value=-1)),
            ("objetivo", CodificadorPorObjetivo(smoothing=suavizado)),
        ])
    else:
        raise ValueError(f"Estrategia de codificación no reconocida: {codificacion}")

    bloques = []
    if num:
        bloques.append(("num", t_num, num))
    if baja:
        bloques.append(("cat_baja", t_baja, baja))
    if alta:
        bloques.append(("cat_alta", t_alta, alta))
    if tec:
        bloques.append(("tecnologia", "passthrough", tec))

    return ColumnTransformer(bloques, remainder="drop", verbose_feature_names_out=False)


def nombres_de_variables(preprocesador: ColumnTransformer) -> list[str]:
    """Nombres de las columnas transformadas, para SHAP."""
    try:
        return list(preprocesador.get_feature_names_out())
    except Exception:
        # La codificación por objetivo devuelve una columna por variable, así
        # que valen los nombres originales.
        nombres = []
        for nombre, transformador, columnas in preprocesador.transformers_:
            if transformador == "passthrough":
                nombres.extend(columnas)
            elif hasattr(transformador, "get_feature_names_out"):
                try:
                    nombres.extend(transformador.get_feature_names_out(columnas))
                except Exception:
                    nombres.extend(columnas)
            else:
                nombres.extend(columnas)
        return nombres


def preparar_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Separa predictores (X), objetivo (y) y atributos de auditoría (A).

    El género no entra al modelo pero se conserva para auditar. País, renta y
    edad sí son predictores (la geografía es el factor de más peso) y aun así
    se auditan: el modelo puede servir peor a un grupo sin que la métrica
    global lo muestre.
    """
    usadas = (columnas_disponibles(df, NUMERICAS)
              + columnas_disponibles(df, CAT_BAJA + CAT_IA)
              + columnas_disponibles(df, CAT_ALTA)
              + columnas_tecnologia(df))
    X = df[usadas].copy()
    y = df[OBJETIVO].to_numpy()

    dimensiones_auditoria = ["income_group", "wb_region", "Country", "Age",
                             "EdLevel", "OrgSize"]
    if "Gender" in df.columns:
        dimensiones_auditoria.append("Gender")
    A = df[columnas_disponibles(df, dimensiones_auditoria)].copy()

    return X, y, A
