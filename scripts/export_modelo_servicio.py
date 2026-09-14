#!/usr/bin/env python3
"""Exporta el modelo y su contexto para el servicio en línea.

El servicio no entrena ni vuelve a leer la encuesta: recibe una tubería ya
ajustada y un archivo de contexto con todo lo que necesita para responder y,
sobre todo, para saber cuándo **no** debe responder.

El contexto incluye, además de los catálogos de la interfaz, las tres medidas
en que se apoya la compuerta de fiabilidad: cuántas observaciones respaldan cada
combinación de país y rol, el error relativo auditado de cada nivel de renta, y
la dispersión salarial dentro de cada país. Se exportan aquí y no se calculan en
el servidor para que lo que se sirve sea exactamente lo que la tesis documenta.

Uso:
    python scripts/export_modelo_servicio.py
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.data_loader_stackoverflow import TARGET, cargar_encuesta  # noqa: E402
from src.feature_pipeline_so import (columnas_tecnologia,  # noqa: E402
                                     construir_preprocesador,
                                     nombres_de_variables, preparar_xy)
from src.metrics import compute_metrics  # noqa: E402
from src.model_registry import build_models  # noqa: E402

SALIDA = RAIZ / "servicio" / "artefactos"
ANIO = "2023"
SEMILLA = 42
MODELO = "XGBoost"
CODIFICACION = "target"

# La interfaz se presenta en español: los países, roles y categorías del
# formulario se traducen en la etiqueta y conservan el valor original de la
# encuesta, que es el que el modelo entiende.
PAIS_ES = {
    "United States of America": "Estados Unidos",
    "United Kingdom of Great Britain and Northern Ireland": "Reino Unido",
    "Germany": "Alemania", "Canada": "Canadá", "India": "India",
    "France": "Francia", "Netherlands": "Países Bajos", "Australia": "Australia",
    "Brazil": "Brasil", "Spain": "España", "Poland": "Polonia",
    "Sweden": "Suecia", "Italy": "Italia", "Switzerland": "Suiza",
    "Denmark": "Dinamarca", "Norway": "Noruega", "Austria": "Austria",
    "Portugal": "Portugal", "Israel": "Israel", "Finland": "Finlandia",
    "Belgium": "Bélgica", "Czech Republic": "Chequia",
    "New Zealand": "Nueva Zelanda", "Russian Federation": "Rusia",
    "Greece": "Grecia", "Turkey": "Turquía", "Romania": "Rumanía",
    "Ukraine": "Ucrania", "South Africa": "Sudáfrica", "Mexico": "México",
    "Ireland": "Irlanda", "Hungary": "Hungría", "Argentina": "Argentina",
    "Bulgaria": "Bulgaria", "Colombia": "Colombia", "Pakistan": "Pakistán",
    "Japan": "Japón", "Lithuania": "Lituania", "Serbia": "Serbia",
    "Iran, Islamic Republic of...": "Irán", "China": "China",
    "Croatia": "Croacia", "Bangladesh": "Bangladés", "Singapore": "Singapur",
    "Estonia": "Estonia", "Slovenia": "Eslovenia", "Slovakia": "Eslovaquia",
    "Chile": "Chile", "Indonesia": "Indonesia", "Philippines": "Filipinas",
    "Malaysia": "Malasia", "Viet Nam": "Vietnam", "Latvia": "Letonia",
    "Taiwan": "Taiwán", "Hong Kong (S.A.R.)": "Hong Kong",
    "Thailand": "Tailandia", "Georgia": "Georgia",
    "United Arab Emirates": "Emiratos Árabes Unidos",
    "South Korea": "Corea del Sur", "Sri Lanka": "Sri Lanka",
    "Nigeria": "Nigeria", "Egypt": "Egipto", "Uruguay": "Uruguay",
    "Peru": "Perú", "Costa Rica": "Costa Rica", "Armenia": "Armenia",
    "Luxembourg": "Luxemburgo",
    "Venezuela, Bolivarian Republic of...": "Venezuela", "Nepal": "Nepal",
    "Bosnia and Herzegovina": "Bosnia y Herzegovina", "Cyprus": "Chipre",
    "Ecuador": "Ecuador", "Kenya": "Kenia", "Morocco": "Marruecos",
    "Republic of Korea": "Corea (Rep.)", "Belarus": "Bielorrusia",
    "Kazakhstan": "Kazajistán", "Dominican Republic": "República Dominicana",
}

ROL_ES = {
    "Developer, full-stack": "Desarrollo full-stack",
    "Developer, back-end": "Desarrollo back-end",
    "Developer, front-end": "Desarrollo front-end",
    "Developer, desktop or enterprise applications": "Aplicaciones de escritorio o corporativas",
    "Developer, embedded applications or devices": "Sistemas embebidos",
    "Developer, mobile": "Desarrollo móvil",
    "Developer, game or graphics": "Videojuegos o gráficos",
    "Developer, QA or test": "Aseguramiento de calidad",
    "Data scientist or machine learning specialist": "Ciencia de datos / aprendizaje automático",
    "Data or business analyst": "Análisis de datos o de negocio",
    "Engineer, data": "Ingeniería de datos",
    "Engineer, site reliability": "Ingeniería de confiabilidad (SRE)",
    "DevOps specialist": "Especialista DevOps",
    "Cloud infrastructure engineer": "Infraestructura en la nube",
    "Engineering manager": "Dirección de ingeniería",
    "Product manager": "Gestión de producto",
    "Project manager": "Gestión de proyectos",
    "Academic researcher": "Investigación académica",
    "Research & Development role": "Investigación y desarrollo",
    "Scientist": "Científico/a",
    "System administrator": "Administración de sistemas",
    "Database administrator": "Administración de bases de datos",
    "Security professional": "Seguridad",
    "Blockchain": "Blockchain",
    "Hardware Engineer": "Ingeniería de hardware",
    "Educator": "Docencia",
    "Designer": "Diseño",
    "Marketing or sales professional": "Marketing o ventas",
    "Engineer, other": "Ingeniería, otra",
    "Developer, other": "Desarrollo, otro",
    "Senior Executive (C-Suite, VP, etc.)": "Alta dirección",
    "Developer Experience": "Experiencia de desarrollo",
    "Developer Advocate": "Relaciones con desarrolladores",
    "Other": "Otro",
}

EDU_ES = {
    "Primary/elementary school": "Primaria",
    "Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)": "Secundaria",
    "Some college/university study without earning a degree": "Estudios universitarios sin título",
    "Associate degree (A.A., A.S., etc.)": "Título técnico",
    "Bachelor’s degree (B.A., B.S., B.Eng., etc.)": "Grado universitario",
    "Master’s degree (M.A., M.S., M.Eng., MBA, etc.)": "Maestría",
    "Professional degree (JD, MD, Ph.D, Ed.D, etc.)": "Doctorado o grado profesional",
    "Something else": "Otra formación",
}

SECTOR_ES = {
    "Information Services, IT, Software Development, or other Technology": "Tecnología y software",
    "Financial Services": "Servicios financieros",
    "Manufacturing, Transportation, or Supply Chain": "Manufactura y logística",
    "Retail and Consumer Services": "Comercio y consumo",
    "Healthcare": "Salud",
    "Higher Education": "Educación superior",
    "Insurance": "Seguros",
    "Legal Services": "Servicios legales",
    "Oil & Gas": "Petróleo y gas",
    "Advertising Services": "Publicidad",
    "Wholesale": "Mayorista",
    "Other": "Otro sector",
}

# Etiquetas legibles para la interfaz. La encuesta usa cadenas largas que no
# caben en un desplegable.
LEGIBLE = {
    "Just me - I am a freelancer, sole proprietor, etc.": "Solo yo",
    "2 to 9 employees": "2 a 9",
    "10 to 19 employees": "10 a 19",
    "20 to 99 employees": "20 a 99",
    "100 to 499 employees": "100 a 499",
    "500 to 999 employees": "500 a 999",
    "1,000 to 4,999 employees": "1,000 a 4,999",
    "5,000 to 9,999 employees": "5,000 a 9,999",
    "10,000 or more employees": "10,000 o más",
    "I don’t know": "No lo sé",
    "Remote": "Remoto",
    "Hybrid (some remote, some in-person)": "Híbrido",
    "In-person": "Presencial",
}


def nivel_precios() -> dict[str, float]:
    """Nivel de precios por país, de la tabla de paridad ya versionada."""
    import csv
    ruta = RAIZ / "data" / "reference" / "ppp_factors.csv"
    with ruta.open(encoding="utf-8") as fh:
        return {r["pais"]: float(r["nivel_precios"]) for r in csv.DictReader(fh)}


def catalogo(serie: pd.Series, minimo: int = 1,
             traduccion: dict | None = None) -> list[dict]:
    """Categorías declaradas, con su frecuencia, de mayor a menor."""
    v = serie[serie != "No declarado"].value_counts()
    dic = {**LEGIBLE, **(traduccion or {})}
    return [{"valor": str(k), "etiqueta": dic.get(str(k), str(k)), "n": int(n)}
            for k, n in v.items() if n >= minimo]


def main() -> int:
    print(f"Edición {ANIO}, modelo {MODELO}, codificación «{CODIFICACION}»\n")
    df, _ = cargar_encuesta(anio=ANIO)
    X, y, A = preparar_xy(df)

    X_ent, X_pru, y_ent, y_pru = train_test_split(
        X, y, test_size=0.20, random_state=SEMILLA, stratify=df["income_group"])

    tuberia = Pipeline([
        ("preprocesador", construir_preprocesador(df, CODIFICACION)),
        ("modelo", build_models()[MODELO]),
    ]).fit(X_ent, y_ent)

    metricas = compute_metrics(y_pru, tuberia.predict(X_pru))
    print(f"R² sobre prueba: {metricas['R2']:.4f}   MAE: ${metricas['MAE_USD']:,.0f}")

    SALIDA.mkdir(parents=True, exist_ok=True)
    joblib.dump(tuberia, SALIDA / "modelo.joblib", compress=3)
    tam = (SALIDA / "modelo.joblib").stat().st_size / 1e6
    print(f"modelo.joblib: {tam:.1f} MB")

    # --- Medidas en que se apoya la compuerta de fiabilidad ----------------
    por_pais_rol = df.groupby(["Country", "DevType"]).size()
    por_pais = df["Country"].value_counts()

    # Error relativo auditado de cada nivel de renta, tal como lo mide el
    # Capítulo V. Es la banda que el servicio declara al usuario.
    auditoria = json.loads((RAIZ / "results" / "fairness_income_group_2023.json")
                           .read_text(encoding="utf-8"))
    error_grupo = {g: v["mae_relativo"] for g, v in auditoria["por_grupo"].items()}

    dispersion = (df.groupby("income_group", observed=True)
                    .apply(lambda g: g.groupby("Country")["salary_log"].std().median(),
                           include_groups=False).to_dict())

    contexto = {
        "modelo": {
            "algoritmo": MODELO, "codificacion": CODIFICACION, "edicion": ANIO,
            "semilla": SEMILLA,
            "r2_prueba": round(float(metricas["R2"]), 4),
            "mae_usd": int(metricas["MAE_USD"]),
            "n_muestra": int(len(df)),
            "n_paises": int(df["Country"].nunique()),
            "n_entrenamiento": int(len(X_ent)),
            "generado": date.today().isoformat(),
            "python": platform.python_version(),
        },
        "columnas": list(X.columns),
        # Nombres tras la transformación. Se exportan porque el
        # ColumnTransformer no siempre puede deducirlos, y sin ellos el
        # servicio no podría agrupar las contribuciones SHAP por variable.
        "columnas_transformadas": nombres_de_variables(tuberia.named_steps["preprocesador"]),
        "tecnologia": columnas_tecnologia(df),
        "catalogos": {
            "Country": catalogo(df["Country"], traduccion=PAIS_ES),
            "DevType": catalogo(df["DevType"], minimo=30, traduccion=ROL_ES),
            "EdLevel": catalogo(df["EdLevel"], traduccion=EDU_ES),
            "OrgSize": catalogo(df["OrgSize"]),
            "RemoteWork": catalogo(df["RemoteWork"]),
            "Industry": catalogo(df["Industry"], traduccion=SECTOR_ES),
            "ICorPM": catalogo(df["ICorPM"]),
            "Age": catalogo(df["Age"]),
            "lenguajes": sorted(c.replace("language__", "")
                                for c in df.columns if c.startswith("language__")),
        },
        "fiabilidad": {
            "n_pais_rol": {f"{p}||{r}": int(n) for (p, r), n in por_pais_rol.items()},
            "n_pais": {str(k): int(v) for k, v in por_pais.items()},
            "error_relativo_grupo": {k: round(float(v), 4) for k, v in error_grupo.items()},
            "dispersion_intra_pais": {k: round(float(v), 4) for k, v in dispersion.items()},
            "n_minimo_celda": 30,
        },
        "referencia": {
            "income_group": df.drop_duplicates("Country")
                              .set_index("Country")["income_group"].to_dict(),
            # Nivel de precios relativo a EE. UU. (Banco Mundial, 2023), para
            # que la comparación pueda expresarse en poder de compra.
            "nivel_precios": nivel_precios(),
            "mediana_pais": df.groupby("Country")[TARGET].median().round(0).astype(int).to_dict(),
            "mediana_global": int(df[TARGET].median()),
        },
    }
    (SALIDA / "contexto.json").write_text(
        json.dumps(contexto, ensure_ascii=False, indent=1), encoding="utf-8")
    tam = (SALIDA / "contexto.json").stat().st_size / 1e6
    print(f"contexto.json: {tam:.2f} MB · "
          f"{len(contexto['fiabilidad']['n_pais_rol'])} celdas país-rol")

    celdas = pd.Series(contexto["fiabilidad"]["n_pais_rol"])
    print(f"\nCeldas con al menos 30 observaciones: {(celdas >= 30).sum()} de {len(celdas)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
