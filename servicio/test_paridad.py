"""Paridad entre el entorno de entrenamiento y el de ejecución.

El modelo se entrena con Python 3.14 y las versiones del requirements.txt de
la raíz, pero la CPU virtual del servidor no soporta los wheels recientes de
NumPy, así que el servicio corre con Python 3.12 y NumPy 2.0. referencia.json
guarda las estimaciones del entorno de entrenamiento; esto las contrasta al
construir la imagen.
"""

import json
from pathlib import Path

import app as servicio

REFERENCIA = json.loads((Path(__file__).parent / "artefactos" / "referencia.json")
                        .read_text(encoding="utf-8"))

# Un céntimo sobre decenas de miles de dólares: admite redondeo flotante, no
# un cambio de resultado.
TOLERANCIA = 0.01


def test_las_estimaciones_coinciden_con_el_entorno_de_entrenamiento():
    for nombre, caso in REFERENCIA.items():
        fila = servicio.construir_fila(servicio.Perfil(**caso["perfil"]))
        obtenida = servicio.estimar(fila)
        assert abs(obtenida - caso["estimacion"]) < TOLERANCIA, (
            f"{nombre}: el servicio estima {obtenida:,.2f} y el entorno de "
            f"entrenamiento {caso['estimacion']:,.2f}")


def test_las_contribuciones_coinciden():
    for nombre, caso in REFERENCIA.items():
        fila = servicio.construir_fila(servicio.Perfil(**caso["perfil"]))
        obtenidos = {f["factor"]: f["efecto"] for f in servicio.factores(fila)}
        for esperado in caso["factores"]:
            assert esperado["factor"] in obtenidos, f"{nombre}: falta {esperado['factor']}"
            assert abs(obtenidos[esperado["factor"]] - esperado["efecto"]) < 0.001
