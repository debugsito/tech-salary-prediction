#!/usr/bin/env python3
"""Construye el cuaderno reproducible de la tesis.

Genera `notebooks/tesis_reproducible.ipynb` a partir de una especificación
declarativa. El cuaderno se construye y no se edita a mano por dos razones:
los cuadernos editados manualmente acumulan estado oculto y dependencias
implícitas entre celdas —uno de los hallazgos de la auditoría documentada en
§2.1.1—, y su formato JSON hace ilegibles las diferencias en el control de
versiones.

El cuaderno resultante llama a los mismos módulos que los scripts de análisis:
no duplica lógica. Sirve como narración ejecutable del trabajo, no como
implementación alternativa.

Uso:
    python scripts/build_notebook.py
    jupyter nbconvert --execute --to notebook --inplace notebooks/tesis_reproducible.ipynb
"""

import json
from pathlib import Path

MD = "markdown"
CODE = "code"


def celda(tipo, fuente, n=[0]):
    # nbformat 4.5 exige un identificador por celda. Se genera de forma
    # determinista a partir del orden, para que reconstruir el cuaderno no
    # produzca diferencias espurias en el control de versiones.
    n[0] += 1
    base = {"cell_type": tipo, "id": f"celda-{n[0]:02d}",
            "metadata": {}, "source": fuente.strip().split("\n")}
    base["source"] = [l + "\n" for l in base["source"][:-1]] + [base["source"][-1]]
    if tipo == CODE:
        base.update({"execution_count": None, "outputs": []})
    return base


CELDAS = [
(MD, """
# Estimación de compensaciones en el sector tecnológico

**Cuaderno reproducible de la tesis**

Universidad ESAN · Facultad de Ingeniería · Carrera de Ingeniería de Sistemas

Arian Antonio Garay Concha · Carlos Sebastián Ramos Flores

---

Este cuaderno reproduce el experimento completo del Capítulo IV. Llama a los mismos
módulos que los scripts del proyecto, de modo que no existe una implementación
alternativa que pueda divergir de la documentada.

**Requisitos previos:** entorno con las dependencias de `requirements.txt`. Los datos
se descargan automáticamente del repositorio oficial de Stack Overflow (licencia
ODbL 1.0) la primera vez que se ejecuta.

**Duración aproximada:** 8 minutos.

**Semilla:** 42, fijada en partición, inicialización de modelos y remuestreo.
"""),
(CODE, """
import sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
RAIZ = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(RAIZ))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SEMILLA = 42
np.random.seed(SEMILLA)
print("Raíz del proyecto:", RAIZ)
"""),
(MD, """
## 1. Carga y preparación de los datos

El módulo de carga aplica los criterios de inclusión documentados en §3.2.4 y
devuelve, junto al conjunto resultante, el registro del efecto de cada filtro sobre
el tamaño muestral. Ese registro es el que alimenta la tabla del Capítulo IV: no se
transcribe a mano.
"""),
(CODE, """
from src.data_loader_stackoverflow import cargar_encuesta, TARGET

df, registro = cargar_encuesta(anio="2023")
registro
"""),
(CODE, """
print(f"Muestra efectiva: {len(df):,} observaciones × {df.shape[1]} columnas")
print(f"Países: {df['Country'].nunique()}")
print(f"Mediana: ${df[TARGET].median():,.0f}   ·   Media: ${df[TARGET].mean():,.0f}")
print(f"Asimetría: {df[TARGET].skew():.2f}   ·   tras logaritmo: {df['salary_log'].skew():.2f}")
"""),
(MD, """
### 1.1 Distribución de la variable dependiente

La transformación logarítmica reduce la asimetría e invierte su signo. El valor
residual de −0.82 implica que el supuesto de normalidad de los residuos se cumple de
forma aproximada y no exacta, extremo que se declara en §4.1.3.
"""),
(CODE, """
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.2))
a1.hist(df[TARGET] / 1000, bins=60, color="#2a78d6")
a1.set(xlabel="Compensación anual (miles de USD)", ylabel="Frecuencia",
       title=f"Escala original · asimetría {df[TARGET].skew():.2f}")
a2.hist(df["salary_log"], bins=60, color="#2a78d6")
a2.set(xlabel="ln(1 + compensación)",
       title=f"Escala logarítmica · asimetría {df['salary_log'].skew():.2f}")
for a in (a1, a2):
    a.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.show()
"""),
(MD, """
### 1.2 Composición de la muestra

El 85 % de las observaciones corresponde a países de renta alta. Esta representación
desigual condiciona la potencia del análisis de equidad para las regiones de menor
renta, y se declara como limitación en §6.2.1.
"""),
(CODE, """
comp = (df.groupby("income_group", observed=True)[TARGET]
        .agg(N="count", mediana="median")
        .assign(pct=lambda t: (t.N / len(df) * 100).round(1))
        .sort_values("mediana", ascending=False))
comp
"""),
(MD, """
## 2. Diseño experimental

Diseño factorial entre estrategia de representación de variables categóricas de alta
cardinalidad y arquitectura de modelado, evaluado sobre particiones idénticas para
que la comparación admita contraste pareado.

**Nota sobre el número de repeticiones.** La validación cruzada se repite cuatro
veces. El motivo no es aumentar la potencia sino hacer posible el contraste: la
prueba de Wilcoxon con cinco pares tiene un valor *p* mínimo alcanzable de 0.0625,
superior al nivel de significación de 0.05. Con solo cinco particiones ninguna
hipótesis podría aceptarse, con independencia de los datos.
"""),
(CODE, """
from scipy.stats import wilcoxon

for n in (5, 10, 20):
    x = np.arange(n, dtype=float)
    print(f"n = {n:>3} pares  →  valor p mínimo alcanzable = {wilcoxon(x, x + 1).pvalue:.6f}")
"""),
(CODE, """
from src.feature_pipeline_so import preparar_xy, construir_preprocesador
from src.model_registry import build_models
from sklearn.model_selection import train_test_split

X, y, A = preparar_xy(df)
X_ent, X_pru, y_ent, y_pru, A_ent, A_pru = train_test_split(
    X, y, A, test_size=0.20, random_state=SEMILLA, stratify=df["income_group"])

print(f"Predictores: {X.shape[1]}")
print(f"Entrenamiento: {len(X_ent):,}   ·   Prueba: {len(X_pru):,}")
print(f"Modelos disponibles: {', '.join(build_models())}")
"""),
(MD, """
## 3. Entrenamiento y evaluación

La ejecución completa se delega al script del proyecto, que exporta todos los
artefactos. Reproducirla aquí duplicaría la lógica y abriría la posibilidad de que
ambas versiones divergiesen.
"""),
(CODE, """
# Descomentar para reejecutar el experimento completo (unos 4 minutos).
# import subprocess
# subprocess.run([sys.executable, "scripts/run_experiment.py"], cwd=RAIZ, check=True)

cv = pd.read_csv(RAIZ / "results/cv_by_encoding.csv")
resumen = (cv.groupby(["modelo", "codificacion"])[["R2", "MAE_log", "MAE_USD"]]
           .agg(["mean", "std"]).round(4))
resumen.sort_values(("R2", "mean"), ascending=False)
"""),
(MD, """
## 4. Contraste de hipótesis

Los contrastes se agrupan en familias, cada una asociada a la hipótesis que responde.
Comparar todas las configuraciones entre sí obligaría a una corrección por
comparaciones múltiples tan severa que ninguna diferencia real alcanzaría
significación, y además incluiría comparaciones que no responden a ninguna hipótesis.
"""),
(CODE, """
w = pd.read_csv(RAIZ / "results/wilcoxon_tests.csv")
w = w[w.metrica == "R2"]

print("HE1 — estrategia de codificación, a igualdad de modelo\\n")
he1 = w[w.familia == "HE1_codificacion"]
print(he1[["config_a", "config_b", "media_a", "media_b",
           "diferencia", "p_valor", "significativo"]].to_string(index=False))
"""),
(CODE, """
print("Modelo lineal frente a ensamblados\\n")
lin = w[w.familia == "lineal_vs_ensamblado"]
print(lin[["config_a", "config_b", "diferencia", "p_valor", "significativo"]].to_string(index=False))

hg = w[w.familia == "HG_vs_baseline"]
print(f"\\nHG — superioridad sobre la línea base: "
      f"{hg.significativo.sum()}/{len(hg)} significativas, p máximo {hg.p_valor.max():.2e}")
"""),
(MD, """
## 5. Explicabilidad

Se emplea el algoritmo específico para modelos de árbol. El estimador basado en
núcleo resultaría impracticable sobre una muestra de esta magnitud.

Los valores calculados se conservan como artefacto y no solo las figuras derivadas,
para permitir su verificación independiente.
"""),
(CODE, """
shap_res = pd.read_csv(RAIZ / "results/shap_summary_XGBoost_target_2023.csv")
top = shap_res.head(10)[["variable", "shap_medio_abs", "ic_inferior", "ic_superior"]]
top.round(4)
"""),
(CODE, """
t = shap_res.head(12).iloc[::-1]
fig, ax = plt.subplots(figsize=(9, 4))
colores = ["#eb6834" if v == "Country" else "#2a78d6" for v in t.variable]
ax.barh(range(len(t)), t.shap_medio_abs, color=colores)
ax.set_yticks(range(len(t)))
ax.set_yticklabels([v[:38] for v in t.variable], fontsize=8)
ax.set_xlabel("Contribución media, mean(|SHAP|)")
ax.set_title("Determinantes de la compensación estimada", loc="left", fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.show()

pais = float(shap_res.loc[shap_res.variable == "Country", "shap_medio_abs"].iloc[0])
exp = float(shap_res.loc[shap_res.variable == "YearsCodePro_num", "shap_medio_abs"].iloc[0])
print(f"El país aporta {pais/exp:.1f} veces más que los años de experiencia profesional.")
"""),
(MD, """
## 6. Auditoría de equidad

**Precisión determinante.** El error absoluto no es comparable entre grupos con
niveles salariales dispares, y su uso conduce a la conclusión contraria a la
correcta. La celda siguiente lo muestra.
"""),
(CODE, """
import json
aud = json.loads((RAIZ / "results/fairness_income_group_2023.json").read_text())

filas = [{"grupo": g, "N": v["n"],
          "MAE": round(v["mae"]),
          "mediana": round(v["mediana_real"]),
          "MAE/mediana": f"{v['mae_relativo']*100:.1f} %",
          "sesgo": round(v["sesgo_sistematico"])}
         for g, v in aud["por_grupo"].items()]
tabla = pd.DataFrame(filas).sort_values("MAE", ascending=False)
print(tabla.to_string(index=False))

a = aud["agregados"]
print(f"\\nRazón de disparidad ABSOLUTA: {a['razon_disparidad_absoluta']:.2f}"
      f"  →  peor servido: {a['grupo_peor_servido_absoluto']}")
print(f"Razón de disparidad RELATIVA: {a['razon_disparidad_relativa']:.2f}"
      f"  →  peor servido: {a['grupo_peor_servido_relativo']}")
print(f"\\n¿La conclusión se invierte al normalizar? {a['invierte_al_normalizar']}")
"""),
(MD, """
### 6.1 Género: la desigualdad del mercado no es sesgo del modelo

La edición 2022 es la última cuya versión pública incluye la variable de género:
Stack Overflow la retiró de sus conjuntos públicos a partir de 2023.
"""),
(CODE, """
gen = json.loads((RAIZ / "results/fairness_Gender_2022.json").read_text())
for g in ("Man", "Woman", "Non-binary, genderqueer, or gender non-conforming"):
    if g in gen["por_grupo"]:
        v = gen["por_grupo"][g]
        print(f"{g[:32]:34s} N={v['n']:>5,}  error relativo {v['mae_relativo']*100:.1f} %")

c = gen["contrastes"]
print(f"\\n{c['prueba']}: p = {c['p_valor']:.3f}"
      f"  →  {'diferencia significativa' if c['significativo'] else 'sin diferencia significativa'}")
"""),
(MD, """
El modelo estima con la misma exactitud relativa a hombres y a mujeres, **pese a que
la brecha salarial de género está presente en los datos**. Es la distinción central
del trabajo: la desigualdad del mercado y el sesgo del sistema predictivo son
fenómenos distintos.
"""),
(MD, """
## 7. Adopción de inteligencia artificial

La asociación bruta entre uso de herramientas de inteligencia artificial y
compensación es inversa y de magnitud apreciable. La pregunta es si persiste al
condicionar por los determinantes establecidos.
"""),
(CODE, """
ia = pd.read_csv(RAIZ / "results/ia_asociacion.csv")
print(ia[ia.variable == "AISelect"][["categoria", "n", "mediana_usd", "diferencia_pct"]]
      .to_string(index=False))

sh25 = pd.read_csv(RAIZ / "results/shap_summary_XGBoost_target_2025.csv")
suma_ia = sh25[sh25.variable.str.startswith(("AISelect", "AIThreat", "AIAgents",
                                             "LearnCodeAI"))].shap_medio_abs.sum()
pais25 = float(sh25.loc[sh25.variable == "Country", "shap_medio_abs"].iloc[0])
print(f"\\nContribución de las 23 variables de IA: {suma_ia:.4f}")
print(f"Contribución del país:                  {pais25:.4f}")
print(f"El país aporta {pais25/suma_ia:.1f} veces más que todas las variables de IA juntas.")
"""),
(MD, """
La asociación bruta se desvanece al condicionar: constituye un **artefacto de
composición** de la muestra, no un efecto atribuible a la adopción de estas
herramientas.

> El diseño es transversal y las variables son autorreportadas. Este resultado no
> admite lectura causal en ningún sentido: lo que permite afirmar es que la
> diferencia observable se explica por las características de ambos grupos, no que
> la adopción carezca de efecto sobre la compensación.
"""),
(MD, """
## 7. Optimización de hiperparámetros

La búsqueda se realiza **exclusivamente sobre el conjunto de entrenamiento**, con
una validación cruzada interna. Si la selección de la configuración emplease las
observaciones del conjunto de prueba, la evaluación posterior sobre ellas dejaría
de ser independiente.
"""),
(CODE, """
tun = pd.read_csv(RAIZ / "results/tuning_resultados.csv")
print(tun[["modelo", "r2_cv_interna", "r2_prueba", "mae_usd_prueba"]].to_string(index=False))

import json as _json
alpha = _json.loads(tun.loc[tun.modelo == "Ridge", "mejores_parametros"].iloc[0])["alpha"]
print(f"\\nRidge: alpha óptimo = {alpha:.2f}  (valor por defecto: 1.0)")
print("El desempeño no mejora pese a la reconfiguración: la limitación del modelo")
print("lineal es de forma funcional, no de ajuste de hiperparámetros.")
"""),
(MD, """
## 8. Interacciones entre predictores

La importancia por reducción de impureza asigna a cada variable un único número
y **no puede expresar que el efecto de una variable dependa del valor de otra**.
Los valores SHAP sí, al asignar una atribución distinta a cada observación.
"""),
(CODE, """
inter = pd.read_csv(RAIZ / "results/interacciones.csv")
sust = inter[~inter.redundante].head(6)
print("Interacciones sustantivas de mayor fuerza:\\n")
print(sust[["variable", "modulada_por", "fuerza"]].to_string(index=False))
print(f"\\n({int(inter.redundante.sum())} pares descartados por redundancia entre variables)")
"""),
(MD, """
## 9. ¿Puede corregirse la disparidad regional?

El 85 % de las observaciones procede de países de renta alta. Cabe preguntarse si
la disparidad detectada responde a ese desequilibrio.
"""),
(CODE, """
mit = pd.read_csv(RAIZ / "results/mitigacion_resultados.csv")
print(mit[["estrategia", "n_entrenamiento", "R2", "MAE_USD",
           "razon_disparidad_renta"]].to_string(index=False))

grp = pd.read_csv(RAIZ / "results/mitigacion_por_grupo.csv")
print("\\nError relativo por grupo y estrategia (%):\\n")
print(grp.pivot_table(index="grupo", columns="estrategia",
                      values="mae_relativo", observed=True).to_string())
"""),
(MD, """
**El error de los grupos minoritarios empeora ligeramente con todas las
estrategias.** Lo poco que desciende la razón de disparidad no procede de servir
mejor a esos grupos, sino de servir peor al mayoritario.

La causa se encuentra en la propia variable dependiente.
"""),
(CODE, """
disp = (df.groupby("income_group", observed=True)
        .apply(lambda g: g.groupby("Country")["salary_log"].std().median(),
               include_groups=False)
        .rename("desv_tipica_intra_pais").to_frame())
disp["razon_vs_renta_alta"] = (disp.desv_tipica_intra_pais /
                               disp.loc["High income", "desv_tipica_intra_pais"])
print(disp.round(3).to_string())
print("\\nLa dispersión salarial DENTRO de cada país es sustancialmente mayor en los")
print("mercados de menor renta. El salario es allí intrínsecamente menos predecible")
print("con las variables disponibles: la disparidad no responde a un déficit de")
print("representación, sino a heterogeneidad intrínseca del fenómeno.")
"""),
(MD, """
## 10. Síntesis

| Hipótesis | Veredicto |
|---|---|
| **HG** — más exacto que la línea base, explicable y auditable | Se acepta |
| **HE1** — la codificación por objetivo supera a la disyuntiva | **Se rechaza**: el efecto cambia de signo según la familia del modelo |
| **HE2** — el capital humano predice más que lo demográfico | Se acepta, pero el bloque geográfico supera a ambos |
| **HE3** — SHAP revela lo que la impureza no captura | Se acepta (ρ = 0.749) y 8 interacciones documentadas |
| **HE4** — hay disparidades sistemáticas del error | Se acepta parcialmente: regional sí, de género no. Su origen es la heterogeneidad intrínseca, no el desbalance |
| **HE5** — la asociación IA-salario no persiste tras el control | Se acepta |

**Hallazgo principal:** el país concentra una contribución tres veces superior a la de
la experiencia profesional. En el mercado analizado, la ubicación pesa más que la
trayectoria.

---

*Para reproducir el trabajo completo desde cero, véase `docs/REPOSITORIOS.md`.*
"""),
]


def main():
    cuaderno = {
        "cells": [celda(t, f) for t, f in CELDAS],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.14"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    salida = Path("notebooks")
    salida.mkdir(exist_ok=True)
    ruta = salida / "tesis_reproducible.ipynb"
    ruta.write_text(json.dumps(cuaderno, indent=1, ensure_ascii=False), encoding="utf-8")
    n_code = sum(1 for t, _ in CELDAS if t == CODE)
    print(f"Escrito {ruta}: {len(CELDAS)} celdas ({n_code} de código)")


if __name__ == "__main__":
    main()
