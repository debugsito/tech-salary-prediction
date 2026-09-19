#!/usr/bin/env python3
"""Matriz de operacionalización de variables para el capítulo de metodología.
Toma de src/tuberia_caracteristicas.py qué variables entran y con qué papel,
mide escala, cardinalidad y cobertura sobre la muestra, y solo la definición
conceptual va escrita a mano. Se inserta en el documento entre marcas.

Uso:
    python scripts/generar_matriz_operacionalizacion.py [--destino archivo.md]
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.carga_encuesta import CATEGORIA_AUSENTE, TARGET, cargar_encuesta  # noqa: E402
from src.tuberia_caracteristicas import (CAT_ALTA, CAT_BAJA, NUMERICAS,  # noqa: E402
                                     OBJETIVO, columnas_tecnologia)

# En un clon aislado no existe el capítulo; se escribe en resultados/.
_CAPITULO = RAIZ.parent / "tesis_borrador" / "06_capitulo_3_metodologia.md"
DESTINO_POR_DEFECTO = (_CAPITULO if _CAPITULO.exists()
                       else RAIZ / "resultados" / "matriz_operacionalizacion.md")
MARCA_INICIO = "<!-- MATRIZ_OPERACIONALIZACION:inicio -->"
MARCA_FIN = "<!-- MATRIZ_OPERACIONALIZACION:fin -->"

# Qué mide cada variable y de qué constructo procede; el resto sale de la muestra.
CONCEPTO = {
    "YearsCodePro_num": ("Experiencia profesional",
                         "Años ejerciendo la programación de forma remunerada. "
                         "Capital humano específico (Becker, 1964)"),
    "YearsCode_num": ("Antigüedad total programando",
                      "Años programando, incluida la formación previa al ejercicio "
                      "profesional. Capital humano general"),
    "WorkExp_num": ("Experiencia laboral total",
                    "Años de vida laboral, con independencia del sector. Sustituye "
                    "a la experiencia profesional en la edición 2025, que no la recoge"),
    "EdLevel": ("Nivel educativo",
                "Máximo nivel de educación formal alcanzado. Señal de productividad "
                "en el sentido de Spence (1973)"),
    "OrgSize": ("Tamaño de la organización",
                "Número de empleados de la empresa. Aproxima la capacidad de pago y "
                "la estructura interna de remuneración"),
    "RemoteWork": ("Modalidad de trabajo",
                   "Presencial, híbrida o remota. Determina si la retribución se "
                   "ancla al mercado local o a uno más amplio"),
    "Industry": ("Sector de actividad",
                 "Rama económica de la organización empleadora"),
    "ICorPM": ("Orientación del puesto",
               "Distingue la trayectoria técnica de la de gestión de personas"),
    "Age": ("Tramo de edad",
            "Variable demográfica. Se incorpora como predictor y se audita como "
            "dimensión de equidad"),
    "income_group": ("Nivel de renta del país",
                     "Clasificación del Banco Mundial del país de residencia. "
                     "Segmentación de mercados (Doeringer & Piore, 1971)"),
    "Country": ("País de residencia",
                "Mercado laboral en que se percibe la retribución"),
    "DevType": ("Rol desempeñado",
                "Función profesional declarada dentro del desarrollo de software"),
}

TECNOLOGIAS = ("Tecnologías dominadas",
               "Lenguajes, bases de datos y plataformas con que la persona declara "
               "haber trabajado. Capital humano específico de carácter técnico")


def escala(serie: pd.Series, cat: bool) -> str:
    if not cat:
        return f"Razón, en años"
    n = serie[serie != CATEGORIA_AUSENTE].nunique()
    return f"Nominal, {n} categorías"


def cobertura(df: pd.DataFrame, col: str) -> str:
    """Proporción que declara la variable. No basta notna(): el cargador
    rellena las categóricas ausentes con una etiqueta propia."""
    if col not in df.columns:
        return "—"
    s = df[col]
    declarado = s.notna()
    if not pd.api.types.is_numeric_dtype(s):
        declarado &= s.astype("string") != CATEGORIA_AUSENTE
    return f"{declarado.mean() * 100:.1f} %"


def fila(nombre, concepto, origen, esc, transformacion, cob):
    return f"| **{nombre}** | {concepto} | `{origen}` | {esc} | {transformacion} | {cob} |"


def construir(df: pd.DataFrame) -> list[str]:
    l = ["**Tabla 0.0.** Matriz de operacionalización de las variables", "",
         "| Variable | Definición conceptual | Columna en la fuente | Escala | "
         "Tratamiento aplicado | Cobertura |",
         "| :--- | :--- | :--- | :--- | :--- | ---: |"]

    # Dependiente
    l.append(fila("Compensación anual",
                  "Retribución monetaria anual total antes de impuestos, convertida "
                  "a dólares por la propia encuesta. Variable dependiente",
                  TARGET, "Razón, en dólares",
                  f"Logaritmo natural de uno más el valor (`{OBJETIVO}`), y "
                  "criterio de plausibilidad relativo a la mediana nacional "
                  "(sección 3.2.4)",
                  cobertura(df, TARGET)))

    for col in NUMERICAS:
        nombre, concepto = CONCEPTO[col]
        origen = col.replace("_num", "")
        l.append(fila(nombre, concepto, origen, escala(df[col], False),
                      "Conversión de los extremos declarados como texto a magnitud "
                      "numérica: «menos de un año» a 0.5 y «más de 50 años» a 51",
                      cobertura(df, origen)))

    for col in CAT_ALTA:
        nombre, concepto = CONCEPTO[col]
        l.append(fila(nombre, concepto, col, escala(df[col], True),
                      "Alta cardinalidad: se comparan codificación por objetivo y "
                      "codificación disyuntiva (sección 3.6.2)",
                      cobertura(df, col)))

    for col in CAT_BAJA:
        if col not in df.columns:
            continue
        nombre, concepto = CONCEPTO[col]
        origen = col if col != "income_group" else "Country"
        trans = ("Derivada del país mediante la tabla de referencia versionada"
                 if col == "income_group"
                 else "Codificación disyuntiva. La ausencia recibe categoría propia "
                      "y no se imputa")
        l.append(fila(nombre, concepto, origen, escala(df[col], True), trans,
                      cobertura(df, col)))

    tec = columnas_tecnologia(df)
    nombre, concepto = TECNOLOGIAS
    l.append(fila(nombre, concepto,
                  "LanguageHaveWorkedWith, DatabaseHaveWorkedWith, PlatformHaveWorkedWith",
                  f"Binaria, {len(tec)} indicadores",
                  "Expansión de la lista separada por punto y coma en un indicador "
                  "por tecnología", "100.0 %"))

    l += ["", "*Fuente:* generada desde `src/tuberia_caracteristicas.py` y la muestra "
              "efectiva de la edición 2023 mediante "
              "`scripts/generar_matriz_operacionalizacion.py`.", ""]
    return l


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--destino", type=Path, default=DESTINO_POR_DEFECTO,
                    help="archivo sobre el que escribir la matriz")
    args = ap.parse_args()
    destino = args.destino

    df, _ = cargar_encuesta(anio="2023")
    bloque = "\n".join([MARCA_INICIO, ""] + construir(df) + [MARCA_FIN])

    # Con marcas se sustituye solo el bloque; sin ellas, el archivo entero.
    if destino.exists() and MARCA_INICIO in destino.read_text(encoding="utf-8"):
        texto = destino.read_text(encoding="utf-8")
        inicio = texto.index(MARCA_INICIO)
        fin = texto.index(MARCA_FIN) + len(MARCA_FIN)
        destino.write_text(texto[:inicio] + bloque + texto[fin:], encoding="utf-8")
    else:
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_text(bloque + "\n", encoding="utf-8")
    print(f"Matriz escrita en {destino.name} "
          f"({len(bloque.splitlines())} líneas)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
