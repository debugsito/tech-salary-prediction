"""Carga de la Stack Overflow Developer Survey y criterios de inclusión.

Devuelve el conjunto filtrado y un registro con lo que descarta cada filtro.
No imputa nada y no colapsa las categorías de Country ni DevType: la
cardinalidad alta es justo lo que se estudia después.

Uso:
    df, registro = cargar_encuesta(anio="2023")
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

# Las rutas por defecto se resuelven contra la raíz del proyecto, no contra el
# directorio de trabajo, para que funcione igual desde un cuaderno.
RAIZ = Path(__file__).resolve().parent.parent


def _ruta(relativa: str | Path) -> Path:
    """Resuelve una ruta relativa contra la raíz del proyecto."""
    p = Path(relativa)
    return p if p.is_absolute() or p.exists() else RAIZ / p

TARGET = "ConvertedCompYearly"

N_MIN_PAIS = 30

# Límites de compensación como múltiplos de la mediana del país. Los errores
# típicos son de escala (sueldo mensual en vez de anual, coma corrida), y un
# umbral absoluto en dólares descartaría respuestas legítimas de renta baja.
PLAUSIBILIDAD = (0.20, 6.0)

# Edad mínima plausible para empezar a programar. Con 12 se iban 527 respuestas
# que no tienen por qué ser erróneas.
EDAD_MIN_INICIO = 8

# Extremo superior de cada tramo de edad.
TRAMO_EDAD_MAX = {
    "Under 18 years old": 18, "18-24 years old": 24, "25-34 years old": 34,
    "35-44 years old": 44, "45-54 years old": 54, "55-64 years old": 64,
    "65 years or older": 99,
}

# Se conservan las que existan en la edición; el esquema cambia entre años.
COLUMNAS = [
    "ResponseId", TARGET, "Country", "Employment", "MainBranch",
    "YearsCode", "YearsCodePro", "WorkExp", "EdLevel", "DevType",
    "OrgSize", "RemoteWork", "Industry", "ICorPM", "Age",
    "LanguageHaveWorkedWith", "DatabaseHaveWorkedWith", "PlatformHaveWorkedWith",
    "WebframeHaveWorkedWith",
    # Atributos protegidos, solo hasta 2022.
    "Gender", "Ethnicity", "Trans", "Sexuality", "Accessibility", "MentalHealth",
    # Preguntas sobre inteligencia artificial, desde 2025.
    "AISelect", "AIThreat", "AIAgents", "LearnCodeAI", "AIAcc", "AIComplex",
]

# Textos que la encuesta usa como "sin respuesta" y pandas no ve como nulos.
NO_RESPUESTA = {"NA", "Prefer not to say", "I prefer not to say", ""}

CATEGORIA_AUSENTE = "No declarado"


# ---------------------------------------------------------------------------
# Descarga

def descargar(anio: str, data_dir: str = "datos/descargas") -> Path:
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
    """Años declarados a número.

    Los extremos vienen como texto: 'Less than 1 year' pasa a 0.5 (cero
    significaría sin experiencia) y 'More than 50 years' a 51.
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


def cargar_referencia_paises(ruta: str = "datos/referencia/grupos_renta_paises.csv") -> pd.DataFrame:
    p = _ruta(ruta)
    if not p.exists():
        raise FileNotFoundError(
            f"No existe {p}. Generar primero con: python scripts/construir_referencia_paises.py")
    return pd.read_csv(p)


def derivar_region(df: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """Añade nivel de renta y región del Banco Mundial.

    Se agrupa por renta y no por continente, como en Prakash y Yadav (2025).
    """
    mapa_renta = dict(zip(ref["country_survey"], ref["income_group"]))
    mapa_region = dict(zip(ref["country_survey"], ref["wb_region"]))
    df["income_group"] = df["Country"].map(mapa_renta).fillna("No clasificado")
    df["wb_region"] = df["Country"].map(mapa_region).fillna("No clasificado")
    return df


def expandir_multivalor(df: pd.DataFrame, columna: str, min_frec: int = 100,
                        prefijo: str | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Columna multivalor (separada por ';') a indicadores binarios.

    Solo las categorías con al menos `min_frec` apariciones; el resto se
    agrupa en un indicador 'otras'.
    """
    if columna not in df.columns:
        return df, []
    prefijo = prefijo or columna.replace("HaveWorkedWith", "").lower()
    listas = df[columna].fillna("").str.split(";")
    frecuencias = listas.explode().str.strip().value_counts()
    frecuentes = [v for v in frecuencias[frecuencias >= min_frec].index if v]

    conjuntos = listas.apply(lambda xs: {x.strip() for x in xs if x.strip()})
    set_frec = set(frecuentes)

    # Un solo concat: insertar columna a columna fragmenta el DataFrame y tarda mucho más.
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
                    data_dir: str = "datos/descargas",
                    aplicar_filtros: bool = True,
                    expandir_tecnologias: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga una edición y aplica los criterios de inclusión.

    Devuelve el conjunto resultante y el registro de descartes por paso.
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

        # 2. Situación de empleo. Hasta 2023 es "Employed, full-time"; en 2025
        # solo "Employed", sin jornada. Se filtra por prefijo.
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

        # 4. Países con representación suficiente. Va antes del filtro de
        # plausibilidad, que necesita una mediana nacional estable.
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

        # 6. Coherencia interna: más años profesionales que totales, o haber
        # empezado antes de EDAD_MIN_INICIO según el tramo de edad. Son errores
        # de captura, no atípicos.
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

    # Derivaciones. Ninguna imputa: el ausente sigue ausente o recibe categoría propia.
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
    """Resumen por consola: efecto de los filtros y distribución de la compensación."""
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
