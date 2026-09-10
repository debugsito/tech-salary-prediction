"""Auditoría de equidad para problemas de regresión.

Implementa los indicadores definidos en §2.4.3.5 de la tesis, formulados sobre
la variable continua siguiendo a Agarwal et al. (2019). Se descarta la
discretización de la variable dependiente, procedimiento que descarta
información y cuyos resultados dependen del umbral elegido.

Indicadores calculados, sobre la partición inducida por un atributo protegido:

    (a) Error por grupo, MAE_g, en términos absolutos y relativos
    (b) Disparidad absoluta del error
    (c) Razón de disparidad del error
    (d) Coeficiente de variación del error entre grupos
    (e) Pérdida acotada por grupo
    (f) Sesgo sistemático con signo, b_g
    (g) Paridad estadística de las predicciones (Kolmogórov-Smirnov)

Los indicadores (f) y (g) responden a preguntas distintas y no deben
confundirse: el primero detecta si el modelo se equivoca siempre en la misma
dirección para un grupo; el segundo, si estima niveles distintos. Solo el
primero constituye evidencia de comportamiento inequitativo del modelo.

**Error absoluto frente a error relativo.** Cuando los grupos comparados
presentan niveles salariales muy dispares, el error absoluto no es comparable
entre ellos: un error de 27,000 dólares sobre una mediana de 87,000 supone una
desviación del 31 %, mientras que uno de 8,500 sobre una mediana de 17,000
supone el 50 %. Ambas magnitudes conducen a conclusiones opuestas sobre qué
grupo recibe peor servicio. Por ello se calcula el error normalizado por la
mediana del propio grupo, y es esta magnitud —y no la absoluta— la que
fundamenta el veredicto sobre la equidad del modelo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Umbral de disparidad relativa, por analogía con el criterio de los cuatro
# quintos empleado en la evaluación de impacto adverso.
UMBRAL_RAZON = 1.25

# Tamaño mínimo de grupo para el tratamiento inferencial. Por debajo, el grupo
# se reporta pero se marca como descriptivo.
N_MIN_GRUPO = 30


def auditar(y_real, y_pred, atributo: pd.Series, en_escala_log: bool = True,
            umbral_zeta: float | None = None) -> dict:
    """Audita la equidad del modelo sobre la partición inducida por `atributo`.

    Si `en_escala_log` es cierto, los errores se reconvierten a la escala
    original antes de calcular los indicadores: solo en esa escala admiten
    interpretación sustantiva y comparación con las magnitudes del mercado.
    """
    y_real = np.asarray(y_real, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if en_escala_log:
        y_real, y_pred = np.expm1(y_real), np.expm1(y_pred)

    datos = pd.DataFrame({
        "real": y_real,
        "pred": y_pred,
        "grupo": np.asarray(atributo).ravel(),
    }).dropna(subset=["grupo"])

    por_grupo = {}
    for grupo, sub in datos.groupby("grupo", observed=True):
        error = sub["pred"] - sub["real"]
        mediana = float(sub["real"].median())
        mae = float(error.abs().mean())
        por_grupo[str(grupo)] = {
            "n": int(len(sub)),
            "suficiente_para_inferencia": bool(len(sub) >= N_MIN_GRUPO),
            "mae": mae,
            # Error normalizado por la mediana del grupo: es el indicador
            # comparable entre grupos con escalas salariales distintas.
            "mae_relativo": float(mae / mediana) if mediana > 0 else None,
            "mape": float((error.abs() / sub["real"].clip(lower=1)).mean()),
            "rmse": float(np.sqrt((error ** 2).mean())),
            "sesgo_sistematico": float(error.mean()),
            "sesgo_relativo": float(error.mean() / mediana) if mediana > 0 else None,
            "mediana_real": mediana,
            "mediana_pred": float(sub["pred"].median()),
        }

    # Los indicadores agregados se calculan solo sobre grupos con tamaño
    # suficiente: incluir grupos minúsculos produciría razones de disparidad
    # infladas por la varianza muestral y no por el comportamiento del modelo.
    validos = {g: v for g, v in por_grupo.items() if v["suficiente_para_inferencia"]}
    resultado = {
        "n_total": int(len(datos)),
        "n_grupos": len(por_grupo),
        "n_grupos_suficientes": len(validos),
        "por_grupo": por_grupo,
    }

    if len(validos) >= 2:
        maes = np.array([v["mae"] for v in validos.values()])
        rel = np.array([v["mae_relativo"] for v in validos.values()], dtype=float)
        sesgos = np.array([v["sesgo_sistematico"] for v in validos.values()])
        peor_abs = max(validos, key=lambda g: validos[g]["mae"])
        peor_rel = max(validos, key=lambda g: validos[g]["mae_relativo"])
        resultado["agregados"] = {
            # Absolutos: se reportan por transparencia, pero no fundamentan el
            # veredicto cuando los grupos difieren en escala salarial.
            "disparidad_absoluta": float(maes.max() - maes.min()),
            "razon_disparidad_absoluta": float(maes.max() / maes.min()),
            "grupo_peor_servido_absoluto": peor_abs,
            # Relativos: son los que fundamentan el veredicto.
            "razon_disparidad_relativa": float(np.nanmax(rel) / np.nanmin(rel)),
            "coeficiente_variacion_relativo": float(np.nanstd(rel) / np.nanmean(rel)),
            "grupo_peor_servido_relativo": peor_rel,
            "supera_umbral_razon": bool(np.nanmax(rel) / np.nanmin(rel) > UMBRAL_RAZON),
            "invierte_al_normalizar": bool(peor_abs != peor_rel),
            "sesgo_maximo_absoluto": float(np.abs(sesgos).max()),
            "sesgo_todos_mismo_signo": bool(np.all(sesgos < 0) or np.all(sesgos > 0)),
            "rango_sesgo": float(sesgos.max() - sesgos.min()),
        }
        if umbral_zeta is not None:
            resultado["agregados"]["perdida_acotada_cumple"] = bool(maes.max() <= umbral_zeta)
            resultado["agregados"]["umbral_zeta"] = float(umbral_zeta)

    resultado["contrastes"] = _contrastar(datos, validos)
    return resultado


def _contrastar(datos: pd.DataFrame, validos: dict) -> dict:
    """Pruebas de significancia sobre las disparidades observadas.

    Mann-Whitney para dos grupos y Kruskal-Wallis para más de dos, sobre el
    error con signo. Se emplean pruebas no paramétricas porque la distribución
    del error no es normal (§4.1.3). Se añade la distancia de Kolmogórov-Smirnov
    entre las distribuciones predichas de los dos grupos extremos.
    """
    from scipy.stats import kruskal, ks_2samp, mannwhitneyu

    grupos = list(validos)
    if len(grupos) < 2:
        return {}

    errores = {g: (datos.loc[datos.grupo.astype(str) == g, "pred"]
                   - datos.loc[datos.grupo.astype(str) == g, "real"]).to_numpy()
               for g in grupos}

    salida = {}
    if len(grupos) == 2:
        a, b = grupos
        u, p = mannwhitneyu(errores[a], errores[b], alternative="two-sided")
        salida["prueba"] = "Mann-Whitney"
        salida["estadistico"] = float(u)
        salida["p_valor"] = float(p)
    else:
        h, p = kruskal(*errores.values())
        salida["prueba"] = "Kruskal-Wallis"
        salida["estadistico"] = float(h)
        salida["p_valor"] = float(p)
    salida["significativo"] = bool(salida["p_valor"] < 0.05)

    # Paridad estadística: distancia entre las distribuciones predichas de los
    # grupos con mediana estimada más alta y más baja.
    medianas = {g: validos[g]["mediana_pred"] for g in grupos}
    alto = max(medianas, key=medianas.get)
    bajo = min(medianas, key=medianas.get)
    if alto != bajo:
        pred_alto = datos.loc[datos.grupo.astype(str) == alto, "pred"]
        pred_bajo = datos.loc[datos.grupo.astype(str) == bajo, "pred"]
        ks = ks_2samp(pred_alto, pred_bajo)
        salida["paridad_estadistica"] = {
            "grupo_prediccion_alta": alto,
            "grupo_prediccion_baja": bajo,
            "distancia_ks": float(ks.statistic),
            "p_valor": float(ks.pvalue),
        }
    return salida


def resumir(auditoria: dict, titulo: str = "") -> str:
    """Informe legible de una auditoría, para registro en el documento."""
    l = [f"Auditoría de equidad — {titulo}" if titulo else "Auditoría de equidad",
         f"  N = {auditoria['n_total']:,} · grupos: {auditoria['n_grupos']} "
         f"({auditoria['n_grupos_suficientes']} con tamaño suficiente)", ""]
    l.append(f"  {'grupo':26s} {'N':>6s} {'MAE':>10s} {'MAE/mediana':>12s} {'sesgo':>11s}")
    for g, v in sorted(auditoria["por_grupo"].items(),
                       key=lambda kv: -(kv[1]["mae_relativo"] or 0)):
        marca = "" if v["suficiente_para_inferencia"] else "  (descriptivo)"
        rel = f"{v['mae_relativo']*100:>11.1f} %" if v["mae_relativo"] else "          —"
        l.append(f"  {g[:26]:26s} {v['n']:>6,} {v['mae']:>10,.0f} {rel} "
                 f"{v['sesgo_sistematico']:>+11,.0f}{marca}")

    a = auditoria.get("agregados")
    if a:
        l += ["",
              f"  Razón de disparidad RELATIVA .. {a['razon_disparidad_relativa']:.3f}"
              f"   {'SUPERA el umbral de 1.25' if a['supera_umbral_razon'] else 'dentro del umbral'}",
              f"  Peor servido (relativo) ....... {a['grupo_peor_servido_relativo']}",
              f"  Coef. de variación relativo ... {a['coeficiente_variacion_relativo']:.3f}",
              "",
              f"  [Referencia] razón absoluta ... {a['razon_disparidad_absoluta']:.3f}"
              f"  · peor servido: {a['grupo_peor_servido_absoluto']}"]
        if a["invierte_al_normalizar"]:
            l.append("  ⚠ El grupo peor servido cambia al normalizar por la escala "
                     "salarial:\n    el error absoluto no es comparable entre estos grupos.")
        l.append(f"  Sesgo máximo absoluto ......... ${a['sesgo_maximo_absoluto']:,.0f}"
                 + ("  · todos los grupos en la misma dirección"
                    if a["sesgo_todos_mismo_signo"] else ""))

    c = auditoria.get("contrastes")
    if c and "p_valor" in c:
        l += ["", f"  {c['prueba']}: p = {c['p_valor']:.3e}"
              f"   {'diferencia significativa' if c['significativo'] else 'sin diferencia significativa'}"]
        pe = c.get("paridad_estadistica")
        if pe:
            l.append(f"  Paridad estadística (KS): D = {pe['distancia_ks']:.3f} "
                     f"entre «{pe['grupo_prediccion_alta'][:20]}» y «{pe['grupo_prediccion_baja'][:20]}»")
    return "\n".join(l)


# Nombres anteriores, conservados para no romper llamadas existentes.
audit_fairness = auditar
get_fairness_summary = resumir
