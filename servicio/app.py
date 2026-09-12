"""Servicio de estimación salarial explicable con compuerta de fiabilidad.

Tres vistas sobre un mismo motor: la del candidato, que quiere saber qué mueve
su cifra; la del reclutador, que necesita una banda que pueda defender; y la
comparación entre mercados, que es donde el peso de la geografía se ve.

No guarda nada. Cada petición se responde y se olvida: no hay base de datos, no
hay sesión y no se registra el perfil consultado, de modo que el servicio no
trata datos personales.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import fiabilidad

RAIZ = Path(__file__).resolve().parent
ARTEFACTOS = RAIZ / "artefactos"

# La tubería serializada referencia `src.encoders.MeanTargetEncoder`, que es la
# codificación por objetivo del proyecto. Hay que poder importarla para
# deserializarla: en la imagen el paquete viaja junto a este archivo; al
# ejecutar desde el repositorio está un nivel por encima.
sys.path.insert(0, str(RAIZ if (RAIZ / "src").exists() else RAIZ.parent))

CONTEXTO = json.loads((ARTEFACTOS / "contexto.json").read_text(encoding="utf-8"))
TUBERIA = joblib.load(ARTEFACTOS / "modelo.joblib")
PREPROCESADOR = TUBERIA.named_steps["preprocesador"]
MODELO = TUBERIA.named_steps["modelo"]
EXPLICADOR = shap.TreeExplainer(MODELO)

COLUMNAS = CONTEXTO["columnas"]
TECNOLOGIA = set(CONTEXTO["tecnologia"])
NUMERICAS = {"YearsCode_num", "YearsCodePro_num", "WorkExp_num"}

NOMBRES = CONTEXTO["columnas_transformadas"]

# Cómo se agrupan las columnas transformadas para presentarlas. Una categórica
# se abre en muchas columnas, y mostrarlas sueltas no diría nada: se suma la
# contribución de todas las que pertenecen a la misma variable.
GRUPOS = [
    ("País de residencia", ("Country", "income_group", "wb_region")),
    ("Experiencia", ("YearsCodePro_num", "YearsCode_num", "WorkExp_num")),
    ("Rol desempeñado", ("DevType",)),
    ("Tecnologías", ("language__", "database__", "platform__", "webframe__")),
    ("Tamaño de empresa", ("OrgSize",)),
    ("Nivel educativo", ("EdLevel",)),
    ("Modalidad de trabajo", ("RemoteWork",)),
    ("Sector de actividad", ("Industry",)),
    ("Orientación del puesto", ("ICorPM",)),
    ("Edad", ("Age",)),
]


class Perfil(BaseModel):
    pais: str
    rol: str | None = None
    anios_profesionales: float | None = Field(default=None, ge=0, le=60)
    anios_programando: float | None = Field(default=None, ge=0, le=60)
    educacion: str | None = None
    tamano_empresa: str | None = None
    modalidad: str | None = None
    sector: str | None = None
    orientacion: str | None = None
    edad: str | None = None
    lenguajes: list[str] = Field(default_factory=list)


class Comparacion(BaseModel):
    perfil: Perfil
    paises: list[str] = Field(default_factory=list, max_length=8)


def construir_fila(p: Perfil) -> pd.DataFrame:
    """Traduce el perfil del formulario a la fila que el modelo espera.

    Lo no declarado se deja como tal: las numéricas quedan ausentes, para que
    las impute el mismo transformador que se ajustó, y las categóricas reciben
    la etiqueta explícita que usaron en el entrenamiento.
    """
    fila = {}
    for col in COLUMNAS:
        if col in TECNOLOGIA:
            fila[col] = 0
        elif col in NUMERICAS:
            fila[col] = np.nan
        else:
            fila[col] = "No declarado"

    fila["Country"] = p.pais
    if "income_group" in fila:
        fila["income_group"] = CONTEXTO["referencia"]["income_group"].get(
            p.pais, "No declarado")

    for clave, col in (("rol", "DevType"), ("educacion", "EdLevel"),
                       ("tamano_empresa", "OrgSize"), ("modalidad", "RemoteWork"),
                       ("sector", "Industry"), ("orientacion", "ICorPM"),
                       ("edad", "Age")):
        valor = getattr(p, clave)
        if valor and col in fila:
            fila[col] = valor

    if p.anios_profesionales is not None:
        fila["YearsCodePro_num"] = p.anios_profesionales
        # Quien no declara la antigüedad total lleva programando al menos su
        # vida profesional: se toma esa como cota inferior en lugar de imputar.
        if p.anios_programando is None:
            fila["YearsCode_num"] = p.anios_profesionales
    if p.anios_programando is not None:
        fila["YearsCode_num"] = p.anios_programando

    for lenguaje in p.lenguajes:
        col = f"language__{lenguaje}"
        if col in fila:
            fila[col] = 1

    return pd.DataFrame([fila], columns=COLUMNAS)


def estimar(fila: pd.DataFrame) -> float:
    return float(np.expm1(TUBERIA.predict(fila)[0]))


def factores(fila: pd.DataFrame, limite: int = 7) -> list[dict]:
    """Contribución de cada bloque de variables a esta estimación concreta."""
    if not NOMBRES:
        return []
    X = np.asarray(PREPROCESADOR.transform(fila), dtype=float)
    valores = np.asarray(EXPLICADOR.shap_values(X)).ravel()
    if len(valores) != len(NOMBRES):
        return []

    acumulado: dict[str, float] = {}
    for nombre, v in zip(NOMBRES, valores):
        etiqueta = "Otros"
        for titulo, prefijos in GRUPOS:
            if any(nombre.startswith(pre) for pre in prefijos):
                etiqueta = titulo
                break
        acumulado[etiqueta] = acumulado.get(etiqueta, 0.0) + float(v)

    orden = sorted(acumulado.items(), key=lambda kv: abs(kv[1]), reverse=True)
    return [{"factor": k, "efecto": round(v, 4),
             "sentido": "sube" if v > 0 else "baja"}
            for k, v in orden[:limite] if abs(v) > 1e-4]


app = FastAPI(title="Estimación salarial explicable", docs_url="/api/docs")


@app.get("/api/salud")
def salud():
    return {"estado": "ok", "modelo": CONTEXTO["modelo"]}


@app.get("/api/contexto")
def contexto():
    """Catálogos del formulario y ficha del modelo."""
    return {
        "modelo": CONTEXTO["modelo"],
        "catalogos": CONTEXTO["catalogos"],
        "umbral_zeta": round(
            fiabilidad.umbral_zeta(CONTEXTO["fiabilidad"]["error_relativo_grupo"]), 4),
        "error_relativo_grupo": CONTEXTO["fiabilidad"]["error_relativo_grupo"],
    }


@app.post("/api/estimar")
def api_estimar(p: Perfil):
    evaluacion = fiabilidad.evaluar(p.pais, p.rol, CONTEXTO)
    respuesta = {
        "fiabilidad": evaluacion,
        "mediana_pais": CONTEXTO["referencia"]["mediana_pais"].get(p.pais),
        "banda": None, "factores": [],
    }
    if not evaluacion["entregar"]:
        return respuesta

    fila = construir_fila(p)
    valor = estimar(fila)
    respuesta["banda"] = fiabilidad.banda(valor, evaluacion)
    respuesta["factores"] = factores(fila)
    return respuesta


@app.post("/api/comparar")
def api_comparar(c: Comparacion):
    if not c.paises:
        raise HTTPException(400, "Indica al menos un país con el que comparar.")
    salida = []
    for pais in dict.fromkeys([c.perfil.pais, *c.paises]):
        p = c.perfil.model_copy(update={"pais": pais})
        evaluacion = fiabilidad.evaluar(pais, p.rol, CONTEXTO)
        fila = {"pais": pais, "fiabilidad": evaluacion,
                "mediana_pais": CONTEXTO["referencia"]["mediana_pais"].get(pais),
                "banda": None}
        if evaluacion["entregar"]:
            fila["banda"] = fiabilidad.banda(estimar(construir_fila(p)), evaluacion)
        salida.append(fila)
    return {"resultados": salida}


app.mount("/static", StaticFiles(directory=RAIZ / "static"), name="static")


@app.get("/")
def inicio():
    return FileResponse(RAIZ / "static" / "index.html")
