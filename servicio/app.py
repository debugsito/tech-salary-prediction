"""API del servicio de estimación salarial con compuerta de fiabilidad.

Estimación con banda y factores SHAP, comparación entre países y catálogos
del formulario. No persiste nada: sin base de datos ni sesión, cada petición
se responde y se olvida.
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

# El joblib referencia src.codificadores. En la imagen el paquete va junto a
# este archivo; desde el repo está un nivel arriba.
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

# Las columnas transformadas se suman por variable de origen para presentarlas;
# una categórica abierta en decenas de columnas no dice nada suelta.
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
    """Perfil del formulario a la fila que espera el modelo.

    Lo no declarado queda ausente (numéricas) o con la etiqueta que se usó en
    el entrenamiento (categóricas); la imputación la hace la tubería.
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
        # Sin antigüedad total, los años profesionales son la cota inferior.
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
    """Contribución SHAP de cada bloque de variables a esta estimación."""
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


@app.get("/api/roles")
def api_roles(pais: str):
    """Roles del país con su respaldo, para que la interfaz ordene y marque el desplegable."""
    fia = CONTEXTO["fiabilidad"]
    minimo = fia["n_minimo_celda"]
    prefijo = f"{pais}||"
    cuenta = {k[len(prefijo):]: v for k, v in fia["n_pais_rol"].items()
              if k.startswith(prefijo)}
    salida = [{"valor": c["valor"], "etiqueta": c["etiqueta"],
               "n": cuenta.get(c["valor"], 0),
               "respaldado": cuenta.get(c["valor"], 0) >= minimo}
              for c in CONTEXTO["catalogos"]["DevType"]]
    salida.sort(key=lambda r: -r["n"])
    return {"pais": pais, "n_pais": fia["n_pais"].get(pais, 0),
            "minimo": minimo, "roles": salida,
            "con_respaldo": sum(1 for r in salida if r["respaldado"])}


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
                # Relativo a EE. UU.; la interfaz lo usa para la banda en poder de compra.
                "nivel_precios": CONTEXTO["referencia"]["nivel_precios"].get(pais),
                "banda": None}
        if evaluacion["entregar"]:
            fila["banda"] = fiabilidad.banda(estimar(construir_fila(p)), evaluacion)
        salida.append(fila)
    return {"resultados": salida}


app.mount("/static", StaticFiles(directory=RAIZ / "static"), name="static")


@app.get("/")
def inicio():
    return FileResponse(RAIZ / "static" / "index.html")
