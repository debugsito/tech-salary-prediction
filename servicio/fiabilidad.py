"""Compuerta de fiabilidad: si se entrega la estimación y con qué reservas.

Tres medidas, todas exportadas del análisis: respaldo de la combinación
país-rol (mismo mínimo de 30 que el criterio de inclusión), pérdida acotada
por grupo sobre el error relativo (Agarwal et al., 2019; el umbral es 1.25
veces el error del grupo mejor servido, no un valor en dólares) y dispersión
salarial dentro del país. Tres niveles; en rojo no se entrega cifra.
"""

from __future__ import annotations

# Razón de disparidad máxima admisible; ver el capítulo 2 de la tesis.
RAZON_MAXIMA = 1.25

VERDE, AMBAR, ROJO = "verde", "ambar", "rojo"


def umbral_zeta(error_por_grupo: dict[str, float]) -> float:
    """Cota de error relativo admisible, a partir del grupo mejor servido."""
    return min(error_por_grupo.values()) * RAZON_MAXIMA


def evaluar(pais: str, rol: str | None, contexto: dict) -> dict:
    """Nivel, motivos y si procede entregar cifra para ese país y rol."""
    fia = contexto["fiabilidad"]
    ref = contexto["referencia"]
    minimo = fia["n_minimo_celda"]

    # Los motivos los lee el usuario: el país va con su nombre en español, no
    # con el valor crudo de la encuesta.
    nombre = next((c["etiqueta"] for c in contexto.get("catalogos", {})
                   .get("Country", []) if c["valor"] == pais), pais)

    n_pais = fia["n_pais"].get(pais, 0)
    n_celda = fia["n_pais_rol"].get(f"{pais}||{rol}", 0) if rol else 0
    grupo = ref["income_group"].get(pais)
    error = fia["error_relativo_grupo"].get(grupo)
    zeta = umbral_zeta(fia["error_relativo_grupo"])

    motivos: list[str] = []

    if n_pais == 0:
        return {
            "nivel": ROJO, "entregar": False,
            "n_pais": 0, "n_celda": 0, "grupo": grupo,
            "error_relativo": error, "umbral_zeta": round(zeta, 4),
            "dispersion_intra_pais": None,
            "motivos": [f"{nombre} no figura en la muestra: la encuesta no reunió "
                        f"las {minimo} respuestas mínimas que exige el estudio."],
            "resumen": "Sin datos para este país.",
        }

    if rol is None:
        # Sin rol no hay criterio de rol que incumplir; se apoya en el país.
        respalda_rol = n_pais >= minimo
        motivos.append(
            f"No se ha indicado el rol, de modo que la estimación se apoya en las "
            f"{n_pais} respuestas de {nombre} sin distinguir la función desempeñada.")
    else:
        respalda_rol = n_celda >= minimo
        if not respalda_rol:
            motivos.append(
                f"Solo {n_celda} de las {n_pais} respuestas de {nombre} corresponden a "
                f"este rol, por debajo del mínimo de {minimo}. La estimación se apoya "
                f"en el país, no en el rol."
                if n_celda else
                f"Ninguna de las {n_pais} respuestas de {nombre} corresponde a este rol.")

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
    """Banda alrededor de la estimación con el error relativo observado en el grupo.

    No es un intervalo de confianza. Se usa el error del grupo y no el global
    porque el global no describe a ningún grupo en particular.
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
