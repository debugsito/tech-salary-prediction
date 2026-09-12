"""Compuerta de fiabilidad: decide si el modelo debe responder, y con qué reservas.

El modelo estima siempre; lo que no siempre existe es el respaldo para esa
estimación. Este módulo lo comprueba antes de que la cifra llegue al usuario, y
cuando no lo hay dice por qué en lugar de devolver un número sin base.

Se apoya en tres medidas, todas exportadas desde el análisis:

1. **Observaciones que respaldan la combinación de país y rol.** Se exige el
   mismo mínimo de treinta que el criterio de inclusión aplica a los países. Por
   debajo, la estimación no se apoya en el rol sino solo en el país.

2. **Pérdida acotada por grupo** (Agarwal et al., 2019), adaptada al error
   relativo. El criterio original acota el error absoluto por grupo, pero un
   umbral en dólares no es comparable entre mercados cuyas medianas difieren en
   un orden de magnitud. Se acota por tanto el error relativo, y el umbral no se
   fija a ojo: es 1.25 veces el del grupo mejor servido, donde 1.25 es la razón
   de disparidad máxima admisible que el marco teórico adopta.

3. **Dispersión salarial dentro del país.** Se informa porque explica el punto
   anterior: donde el salario es intrínsecamente más disperso, ninguna cantidad
   de datos estrecharía la banda.

El resultado tiene tres niveles. En «rojo» el servicio no entrega cifra.
"""

from __future__ import annotations

# Razón de disparidad máxima admisible (sección 2.4.3.5 del documento).
RAZON_MAXIMA = 1.25

VERDE, AMBAR, ROJO = "verde", "ambar", "rojo"


def umbral_zeta(error_por_grupo: dict[str, float]) -> float:
    """Cota de pérdida relativa admisible, derivada del grupo mejor servido."""
    return min(error_por_grupo.values()) * RAZON_MAXIMA


def evaluar(pais: str, rol: str | None, contexto: dict) -> dict:
    """Determina si procede entregar una estimación para ese país y ese rol."""
    fia = contexto["fiabilidad"]
    ref = contexto["referencia"]
    minimo = fia["n_minimo_celda"]

    n_pais = fia["n_pais"].get(pais, 0)
    n_celda = fia["n_pais_rol"].get(f"{pais}||{rol}", 0) if rol else 0
    grupo = ref["income_group"].get(pais)
    error = fia["error_relativo_grupo"].get(grupo)
    zeta = umbral_zeta(fia["error_relativo_grupo"])

    motivos: list[str] = []

    # El país queda fuera de la muestra: no hay nada sobre lo que responder.
    if n_pais == 0:
        return {
            "nivel": ROJO, "entregar": False,
            "n_pais": 0, "n_celda": 0, "grupo": grupo,
            "error_relativo": error, "umbral_zeta": round(zeta, 4),
            "dispersion_intra_pais": None,
            "motivos": [f"{pais} no figura en la muestra: la encuesta no reunió "
                        f"las {minimo} respuestas mínimas que exige el estudio."],
            "resumen": "Sin datos para este país.",
        }

    respalda_rol = n_celda >= minimo
    if not respalda_rol:
        motivos.append(
            f"Solo {n_celda} de las {n_pais} respuestas de {pais} corresponden a "
            f"este rol, por debajo del mínimo de {minimo}. La estimación se apoya "
            f"en el país, no en el rol."
            if n_celda else
            f"Ninguna de las {n_pais} respuestas de {pais} corresponde a este rol.")

    cumple_perdida = error is not None and error <= zeta
    if not cumple_perdida and error is not None:
        motivos.append(
            f"En los países de este nivel de renta el error del modelo equivale al "
            f"{error * 100:.0f} % de la compensación típica, por encima del "
            f"{zeta * 100:.0f} % que el criterio de pérdida acotada admite.")

    dispersion = fia["dispersion_intra_pais"].get(grupo)
    mejor = min(fia["dispersion_intra_pais"].values()) if fia["dispersion_intra_pais"] else None
    if dispersion and mejor and dispersion > mejor * 1.25:
        motivos.append(
            f"El salario varía dentro de un mismo país de este grupo un "
            f"{(dispersion / mejor - 1) * 100:.0f} % más que en los de renta alta, "
            f"de modo que parte de la imprecisión no procede de la falta de datos.")

    if not respalda_rol and not cumple_perdida:
        nivel, entregar = ROJO, False
        resumen = "No hay respaldo suficiente para dar una cifra."
    elif not respalda_rol or not cumple_perdida:
        nivel, entregar = AMBAR, True
        resumen = "Úsala como orden de magnitud, no como referencia cerrada."
    else:
        nivel, entregar = VERDE, True
        resumen = "La estimación se apoya en datos suficientes para este mercado y rol."

    return {
        "nivel": nivel, "entregar": entregar,
        "n_pais": n_pais, "n_celda": n_celda, "grupo": grupo,
        "error_relativo": error, "umbral_zeta": round(zeta, 4),
        "dispersion_intra_pais": dispersion,
        "motivos": motivos, "resumen": resumen,
    }


def banda(estimacion: float, evaluacion: dict) -> dict:
    """Intervalo alrededor de la estimación, con la amplitud medida del grupo.

    No es un intervalo de confianza en sentido estricto: es el error relativo
    que la auditoría observó en el grupo al que pertenece la consulta. Se emplea
    ese y no el error global porque el global no describe a ningún grupo en
    particular, que es precisamente lo que la auditoría demostró.
    """
    e = evaluacion.get("error_relativo")
    if e is None:
        return {"centro": round(estimacion), "inferior": None, "superior": None,
                "amplitud_relativa": None}
    return {
        "centro": round(estimacion),
        "inferior": round(max(0.0, estimacion * (1 - e))),
        "superior": round(estimacion * (1 + e)),
        "amplitud_relativa": round(e, 4),
    }
