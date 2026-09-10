"""Carga y preparación de la Stack Overflow Developer Survey.

Aplica los criterios de inclusión documentados en el Capítulo III de la tesis
(§3.2.4) y devuelve, junto al conjunto resultante, el registro del efecto de
cada filtro sobre el tamaño muestral.

Principios de diseño, derivados de la auditoría documentada en §2.1.1:

1. **No se fabrican valores.** Ninguna columna se rellena con una constante
   inventada, ni se imputa una categoría por defecto cuando el dato falta. La
   ausencia se representa de forma explícita como tal.
2. **No se reduce la cardinalidad.** `Country` y `DevType` se conservan con sus
   categorías originales. Reducirlas a un vocabulario pequeño invalidaría el
   contraste de HE1, cuyo objeto es precisamente la representación de variables
   de alta cardinalidad.
3. **Todo filtro queda registrado.** El número de observaciones descartadas en
   cada paso se devuelve junto a los datos, para que la tabla de §3.2.4 del
   documento se genere a partir de la ejecución y no se transcriba a mano.

Uso:
    from src.data_loader_stackoverflow import cargar_encuesta

    df, registro = cargar_encuesta(anio="2023")
    print(registro.to_string(index=False))
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Fuente

URL_BASE = ("https://github.com/StackExchange/Survey/raw/refs/heads/main/"
            "packages/archive/{anio}/results.csv")

# Raíz del proyecto, deducida de la ubicación de este módulo. Las rutas por
# defecto se resuelven contra ella y no contra el directorio de trabajo, para
# que el módulo funcione igual desde un script, desde un cuaderno o desde
# cualquier otro directorio.
RAIZ = Path(__file__).resolve().parent.parent


def _ruta(relativa: str | Path) -> Path:
    """Resuelve una ruta relativa contra la raíz del proyecto."""
    p = Path(relativa)
    return p if p.is_absolute() or p.exists() else RAIZ / p

TARGET = "ConvertedCompYearly"

# Umbral mínimo de observaciones por país (§3.2.4, criterio 5).
N_MIN_PAIS = 30

# Límites de plausibilidad de la variable dependiente, expresados como múltiplos
# de la mediana del propio país (§3.2.4, criterio 4).
#
# Se emplea una regla relativa al país y no un percentil global por dos razones.
# Primera, los modos de error dominantes son de escala —declarar la retribución
# mensual en lugar de la anual (factor 12), o desplazar el separador decimal
# (factor 10)—, y una regla multiplicativa los captura mientras que un umbral
# absoluto no. Segunda, un umbral absoluto descartaría de forma desproporcionada
# observaciones legítimas de países de renta baja, cuyas medianas nacionales se
# sitúan por debajo del umbral que sería razonable para países de renta alta.
#
# Verificación empírica: frente a la winsorización global en los percentiles 1 y
# 99, esta regla reduce la asimetría de la variable transformada de −1.43 a −0.82
# en la edición 2023, conservando el 95.7 % de las observaciones.
PLAUSIBILIDAD = (0.20, 6.0)

# Edad mínima a la que se considera plausible haber empezado a programar. Por
# debajo, la combinación de tramo de edad y años declarados es incompatible.
# Se fija en 8 y no en un valor mayor porque iniciarse en la programación en la
# infancia es frecuente en esta población: con umbral 12 se descartarían 527
# observaciones que no son necesariamente erróneas.
EDAD_MIN_INICIO = 8

# Extremos superiores de cada tramo de edad, para la comprobación anterior.
TRAMO_EDAD_MAX = {
    "Under 18 years old": 18, "18-24 years old": 24, "25-34 years old": 34,
    "35-44 years old": 44, "45-54 years old": 54, "55-64 years old": 64,
    "65 years or older": 99,
}

# Columnas que se conservan cuando existen en la edición. Las ausentes se
# omiten sin error: el esquema varía entre ediciones (solo 33 campos son
# comunes entre 2022 y 2025) y el diseño lo contempla.
COLUMNAS = [
    "ResponseId", TARGET, "Country", "Employment", "MainBranch",
    "YearsCode", "YearsCodePro", "WorkExp", "EdLevel", "DevType",
    "OrgSize", "RemoteWork", "Industry", "ICorPM", "Age",
    "LanguageHaveWorkedWith", "DatabaseHaveWorkedWith", "PlatformHaveWorkedWith",
    "WebframeHaveWorkedWith",
    # Atributos protegidos: presentes solo hasta la edición 2022.
    "Gender", "Ethnicity", "Trans", "Sexuality", "Accessibility", "MentalHealth",
    # Bloque de inteligencia artificial: presente desde la edición 2025.
    "AISelect", "AIThreat", "AIAgents", "LearnCodeAI", "AIAcc", "AIComplex",
]

# Valores que la encuesta emplea para "sin respuesta" y que pandas no reconoce
# como nulos por tratarse de texto.
NO_RESPUESTA = {"NA", "Prefer not to say", "I prefer not to say", ""}

CATEGORIA_AUSENTE = "No declarado"


# ---------------------------------------------------------------------------
# Descarga

def descargar(anio: str, data_dir: str = "data/external") -> Path:
    """Descarga la edición indicada si no está ya en disco."""
    destino = _ruta(data_dir) / f"so_survey_{anio}.csv"
    if destino.exists():
        return destino
    destino.parent.mkdir(parents=True, exist_ok=True)
    url = URL_BASE.format(anio=anio)
    print(f"Descargando edición {anio} desde {url}")
    urllib.request.urlretrieve(url, destino)
    return destino


# ---------------------------------------------------------------------------
# Conversiones

def parsear_anios(serie: pd.Series) -> pd.Series:
    """Convierte los años declarados a magnitud numérica.

    La encuesta codifica los extremos como texto. 'Less than 1 year' se
    representa como 0.5 y no como 0, porque cero significaría ausencia de
    experiencia y la categoría agrupa a quienes tienen alguna. 'More than
    50 years' se representa como 51, valor que preserva el orden sin sugerir
    una precisión que el dato no tiene.
    """
    s = serie.astype("string").str.strip()
    s = s.replace({
        "Less than 1 year": "0.5",
        "More than 50 years": "51",
        "50 or more years": "51",
    })
    return pd.to_numeric(s, errors="coerce")


def _anios(df: pd.DataFrame, columna: str) -> pd.Series | None:
    """Años declarados en una columna, en forma numérica, o None si no existe."""
    return parsear_anios(df[columna]) if columna in df.columns else None


def normalizar_ausentes(df: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    """Convierte a nulo los textos que la encuesta usa como no respuesta."""
    for c in columnas:
        if c in df.columns:
            df[c] = df[c].replace(list(NO_RESPUESTA), np.nan)
    return df


def cargar_referencia_paises(ruta: str = "data/reference/country_income_groups.csv") -> pd.DataFrame:
    p = _ruta(ruta)
    if not p.exists():
        raise FileNotFoundError(
            f"No existe {p}. Generar primero con: python scripts/build_country_reference.py")
    return pd.read_csv(p)


def derivar_region(df: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """Añade el nivel de renta y la región del Banco Mundial.

    La agrupación por nivel de renta —y no por continente— responde al criterio
    empleado por Prakash y Yadav (2025), cuyos resultados constituyen la
    referencia de contraste de la hipótesis HE4.
    """
    mapa_renta = dict(zip(ref["country_survey"], ref["income_group"]))
    mapa_region = dict(zip(ref["country_survey"], ref["wb_region"]))
    df["income_group"] = df["Country"].map(mapa_renta).fillna("No clasificado")
    df["wb_region"] = df["Country"].map(mapa_region).fillna("No clasificado")
    return df


def expandir_multivalor(df: pd.DataFrame, columna: str, min_frec: int = 100,
                        prefijo: str | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Convierte una columna multivalor separada por ';' en indicadores binarios.

    Solo se conservan las categorías que alcanzan `min_frec` observaciones. Las
    descartadas se agregan en un indicador 'otras', de modo que la información
    de haber declarado tecnologías poco frecuentes no se pierda por completo.
    """
    if columna not in df.columns:
        return df, []
    prefijo = prefijo or columna.replace("HaveWorkedWith", "").lower()
    listas = df[columna].fillna("").str.split(";")
    frecuencias = listas.explode().str.strip().value_counts()
    frecuentes = [v for v in frecuencias[frecuencias >= min_frec].index if v]

    conjuntos = listas.apply(lambda xs: {x.strip() for x in xs if x.strip()})
    set_frec = set(frecuentes)

    # Se construyen todos los indicadores de una vez y se concatenan: insertarlos
    # uno a uno fragmenta el bloque interno de pandas y degrada el rendimiento.
    columnas = {}
    for valor in frecuentes:
        col = f"{prefijo}__{valor.lower().replace(' ', '_').replace('/', '_')}"
        columnas[col] = conjuntos.apply(lambda s, v=valor: int(v in s))
    columnas[f"{prefijo}__otras"] = conjuntos.apply(lambda s: int(bool(s - set_frec)))
    columnas[f"{prefijo}__n"] = conjuntos.apply(len)

    df = pd.concat([df, pd.DataFrame(columnas, index=df.index)], axis=1)
    return df, list(columnas)


# ---------------------------------------------------------------------------
# Carga principal

def cargar_encuesta(anio: str = "2023",
                    data_dir: str = "data/external",
                    aplicar_filtros: bool = True,
                    expandir_tecnologias: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga una edición y aplica los criterios de inclusión de §3.2.4.

    Devuelve el conjunto resultante y un registro con el efecto de cada filtro.
    """
    ruta = descargar(anio, data_dir)
    cabecera = pd.read_csv(ruta, nrows=0)
    presentes = [c for c in COLUMNAS if c in cabecera.columns]
    df = pd.read_csv(ruta, usecols=presentes, low_memory=False)

    registro = [{"paso": 0, "criterio": "Respuestas totales de la edición",
                 "n": len(df), "descartados": 0}]

    df = normalizar_ausentes(df, [c for c in presentes if c != TARGET])

    if aplicar_filtros:
        # 1. La variable dependiente debe estar observada.
        n = len(df)
        df = df[df[TARGET].notna() & (df[TARGET] > 0)]
        registro.append({"paso": 1, "criterio": "Reporta compensación anual",
                         "n": len(df), "descartados": n - len(df)})

        # 2. Situación de empleo.
        #
        # La codificación cambia entre ediciones: hasta 2023 se distingue
        # "Employed, full-time" de la modalidad parcial, mientras que en 2025 la
        # categoría es simplemente "Employed", sin desagregar la jornada. Se
        # aplica un criterio de prefijo común, y la pérdida de granularidad en la
        # edición 2025 se declara como limitación de comparabilidad.
        if "Employment" in df.columns:
            n = len(df)
            emp = df["Employment"].astype("string")
            df = df[emp.str.startswith("Employed", na=False)]
            registro.append({"paso": 2, "criterio": "Situación de empleo: asalariado",
                             "n": len(df), "descartados": n - len(df)})

        # 3. Desarrollador profesional.
        if "MainBranch" in df.columns:
            n = len(df)
            df = df[df["MainBranch"] == "I am a developer by profession"]
            registro.append({"paso": 3, "criterio": "Desarrollador profesional",
                             "n": len(df), "descartados": n - len(df)})

        # 4. Países con representación suficiente.
        #
        # Se aplica antes del filtro de plausibilidad porque este último requiere
        # la mediana nacional, cuya estimación no es estable con pocas observaciones.
        n = len(df)
        frec = df["Country"].value_counts()
        df = df[df["Country"].isin(frec[frec >= N_MIN_PAIS].index)]
        registro.append({
            "paso": 4,
            "criterio": f"País con al menos {N_MIN_PAIS} observaciones",
            "n": len(df), "descartados": n - len(df)})

        # 5. Valores de compensación implausibles.
        n = len(df)
        inf, sup = PLAUSIBILIDAD
        mediana_pais = df.groupby("Country")[TARGET].transform("median")
        df = df[(df[TARGET] >= mediana_pais * inf) & (df[TARGET] <= mediana_pais * sup)]
        registro.append({
            "paso": 5,
            "criterio": f"Compensación dentro de [{inf:g}, {sup:g}] veces la mediana del país",
            "n": len(df), "descartados": n - len(df)})

        # 6. Coherencia interna de las variables declaradas.
        #
        # Se descartan dos combinaciones lógicamente imposibles. La primera es
        # declarar más años de experiencia profesional programando que años
        # totales programando. La segunda, haber empezado a programar antes de
        # una edad mínima plausible, deducida del extremo superior del tramo de
        # edad declarado. En ambos casos se trata de errores de captura, no de
        # observaciones atípicas: no admiten interpretación literal.
        n = len(df)
        coherente = pd.Series(True, index=df.index)
        aos, pro = _anios(df, "YearsCode"), _anios(df, "YearsCodePro")
        if aos is not None and pro is not None:
            coherente &= ~(pro > aos).fillna(False)
        if aos is not None and "Age" in df.columns:
            edad_max = df["Age"].map(TRAMO_EDAD_MAX)
            inicio = edad_max - aos
            coherente &= ~(inicio < EDAD_MIN_INICIO).fillna(False)
        df = df[coherente]
        registro.append({
            "paso": 6,
            "criterio": "Coherencia entre años declarados y tramo de edad",
            "n": len(df), "descartados": n - len(df)})

    df = df.reset_index(drop=True)

    # Derivaciones. Ninguna imputa: los valores ausentes se conservan como tales
    # y las variables categóricas reciben una categoría explícita.
    df = derivar_region(df, cargar_referencia_paises())
    for col in ("YearsCode", "YearsCodePro", "WorkExp"):
        if col in df.columns:
            df[f"{col}_num"] = parsear_anios(df[col])
    df["salary_log"] = np.log1p(df[TARGET])

    for col in ("EdLevel", "DevType", "OrgSize", "RemoteWork", "Industry",
                "ICorPM", "Age", "Gender"):
        if col in df.columns:
            df[col] = df[col].fillna(CATEGORIA_AUSENTE)

    if expandir_tecnologias:
        for col in ("LanguageHaveWorkedWith", "DatabaseHaveWorkedWith",
                    "PlatformHaveWorkedWith"):
            df, _ = expandir_multivalor(df, col)

    return df, pd.DataFrame(registro)


def resumen(df: pd.DataFrame, registro: pd.DataFrame) -> None:
    """Imprime el resumen que alimenta las tablas del Capítulo IV."""
    print("\nEfecto de los criterios de inclusión")
    print(registro.to_string(index=False))
    print(f"\nMuestra efectiva: {len(df):,} observaciones × {df.shape[1]} columnas")
    print(f"Países: {df['Country'].nunique()}")
    print(f"\nCompensación (USD)")
    print(f"  mediana {df[TARGET].median():>12,.0f}")
    print(f"  media   {df[TARGET].mean():>12,.0f}")
    print(f"  asimetría {df[TARGET].skew():>10.2f}   tras log1p {df['salary_log'].skew():>6.2f}")
    print("\nDistribución por nivel de renta")
    for grupo, sub in df.groupby("income_group", observed=True):
        print(f"  {grupo[:28]:30s} {len(sub):>6,}  mediana ${sub[TARGET].median():>9,.0f}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--anio", default="2023")
    ap.add_argument("--sin-filtros", action="store_true")
    args = ap.parse_args()

    datos, reg = cargar_encuesta(anio=args.anio, aplicar_filtros=not args.sin_filtros)
    resumen(datos, reg)
