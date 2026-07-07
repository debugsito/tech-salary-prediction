"""Preprocesamiento de la Stack Overflow Developer Survey.

Construye el transformador de columnas con las dos estrategias de representación
de variables categóricas de alta cardinalidad que la hipótesis HE1 compara:
codificación disyuntiva (*one-hot*) y codificación por objetivo con suavizado.

Sustituye al preprocesador del trabajo previo auditado (§2.1.1), construido
para un esquema cuyas columnas no existen en la encuesta real.

Todos los transformadores que estiman parámetros a partir de los datos —imputación
por mediana, codificación por objetivo, estandarización— se ajustan dentro de la
tubería, de modo que la validación cruzada los reajuste en cada partición y no se
produzca fuga de información (§3.5.3).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.codificadores import CodificadorPorObjetivo

# Variables de alta cardinalidad: son las que HE1 somete a contraste.
CAT_ALTA = ["Country", "DevType"]

# Variables categóricas de baja cardinalidad. Se codifican siempre de forma
# disyuntiva, dado que su número de categorías no plantea el problema de
# dimensionalidad que motiva la codificación por objetivo.
CAT_BAJA = ["EdLevel", "OrgSize", "RemoteWork", "Industry", "ICorPM",
            "Age", "income_group"]

# Variables de adopción y percepción de inteligencia artificial. Solo existen
# en la edición 2025 y se incorporan al modelo para que la atribución SHAP
# pueda estimar su contribución marginal condicionada a los demás predictores,
# que es el criterio de contraste de la hipótesis HE5.
CAT_IA = ["AISelect", "AIThreat", "AIAgents", "LearnCodeAI"]

# Variables numéricas derivadas de los años declarados.
NUMERICAS = ["YearsCode_num", "YearsCodePro_num", "WorkExp_num"]

# Prefijos de los indicadores binarios generados a partir de las columnas
# multivalor de tecnologías.
PREFIJOS_TECNOLOGIA = ("language__", "database__", "platform__")

OBJETIVO = "salary_log"


def columnas_tecnologia(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(PREFIJOS_TECNOLOGIA)]


def columnas_disponibles(df: pd.DataFrame, candidatas: list[str]) -> list[str]:
    """Filtra a las columnas presentes: el esquema varía entre ediciones."""
    return [c for c in candidatas if c in df.columns]


def construir_preprocesador(df: pd.DataFrame, codificacion: str = "target",
                            suavizado: float = 10.0) -> ColumnTransformer:
    """Devuelve el transformador de columnas para la estrategia indicada.

    `codificacion` admite 'target' (por objetivo con suavizado) u 'onehot'
    (disyuntiva completa) para las variables de alta cardinalidad.
    """
    num = columnas_disponibles(df, NUMERICAS)
    baja = columnas_disponibles(df, CAT_BAJA + CAT_IA)
    alta = columnas_disponibles(df, CAT_ALTA)
    tec = columnas_tecnologia(df)

    t_num = Pipeline([
        ("imputador", SimpleImputer(strategy="median")),
        ("escalado", StandardScaler()),
    ])

    # No se imputa la moda en las categóricas: el módulo de carga ya asigna una
    # categoría explícita al valor ausente, porque la ausencia puede ser
    # informativa y la imputación por moda concentraría masa artificialmente
    # en la categoría más frecuente (§3.5.3).
    t_baja = OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                           min_frequency=30)

    if codificacion == "onehot":
        t_alta = OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                               min_frequency=30)
    elif codificacion == "target":
        t_alta = Pipeline([
            # CodificadorPorObjetivo espera códigos numéricos, no etiquetas de texto.
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
    """Nombres de las columnas tras la transformación, para el análisis SHAP."""
    try:
        return list(preprocesador.get_feature_names_out())
    except Exception:
        # La codificación por objetivo devuelve una columna por variable de
        # entrada, de modo que los nombres coinciden con los originales.
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
    """Separa predictores, variable dependiente y atributos de auditoría.

    Se distinguen dos clases de atributo, distinción que el Capítulo III recoge:

    - **Protegidos y excluidos del modelo**: el género. No figura entre los
      predictores pero se conserva para la auditoría posterior.
    - **Dimensiones de auditoría que son a la vez predictores legítimos**: el
      país, el nivel de renta y el tramo de edad. Excluirlos del modelo no
      procedería —la geografía es el determinante salarial de mayor peso, y la
      experiencia se relaciona con la edad—, pero su comportamiento debe
      auditarse igualmente, dado que un modelo puede servir peor a unos grupos
      que a otros sin que ello se aprecie en las métricas agregadas.
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
