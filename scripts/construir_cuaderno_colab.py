#!/usr/bin/env python3
"""Construye el cuaderno reproducible para Google Colab.

Se genera y no se edita a mano: un cuaderno editado acumula estado oculto y
dependencias implícitas entre celdas, que es uno de los defectos que la
auditoría de la sección 2.1.1 encontró en el trabajo previo.

Diferencias con `build_notebook.py`, que produce el cuaderno para ejecutar en
el propio repositorio:

- Clona el proyecto y descarga los datos por su cuenta, de modo que basta abrir
  el archivo en Colab y pulsar «Ejecutar todo».
- No fija las versiones de las bibliotecas. Colab impone las suyas y no coinciden
  con las del Capítulo IV; el cuaderno las anota y comprueba que los resultados
  aguanten el cambio, que es información más útil que un `pip install` que
  fallaría.
- Ejecuta el experimento en lugar de leer los resultados ya calculados, y al
  final contrasta lo obtenido contra las cifras que afirma el documento.

Uso:
    python scripts/construir_cuaderno_colab.py
    jupyter nbconvert --execute --to notebook --inplace notebooks/tesis_colab.ipynb
"""

import json
from pathlib import Path

MD = "markdown"
CODE = "code"
RAIZ = Path(__file__).resolve().parent.parent
DESTINO = RAIZ / "notebooks" / "tesis_colab.ipynb"


def celda(tipo, fuente, n=[0]):
    n[0] += 1
    base = {"cell_type": tipo, "id": f"celda-{n[0]:02d}",
            "metadata": {}, "source": fuente.strip().split("\n")}
    base["source"] = [l + "\n" for l in base["source"][:-1]] + [base["source"][-1]]
    if tipo == CODE:
        base.update({"execution_count": None, "outputs": []})
    return base


CELDAS = [

# ---------------------------------------------------------------------------
(MD, """
# Estimación de compensaciones en el sector tecnológico

**Cuaderno reproducible · Google Colab**

Universidad ESAN · Facultad de Ingeniería · Carrera de Ingeniería de Sistemas

Carlos Sebastian Ramos Flores · Arian Antonio Garay Concha

---

Este cuaderno vuelve a ejecutar el experimento del Capítulo IV desde cero: descarga los
datos, aplica los criterios de inclusión, entrena los modelos, contrasta las hipótesis y
audita la equidad. Al final compara lo que acaba de salir con las cifras que afirma el
documento de tesis.

No reimplementa nada: llama a los mismos módulos del repositorio que usan los scripts de
análisis. Si el cuaderno y el documento discrepan, la discrepancia es real y no un
artefacto de tener dos implementaciones distintas.

**Qué hace falta:** nada. Se abre en Colab y se pulsa «Entorno de ejecución → Ejecutar
todo». Los datos son públicos (Stack Overflow Developer Survey, licencia ODbL 1.0).

**Cuánto tarda:** entre veinte y treinta minutos en modo completo con la CPU que da Colab
gratis. Hay un modo rápido más abajo para quien solo quiera ver que la cadena funciona.
"""),

# ---------------------------------------------------------------------------
(MD, """
## 0. Preparar el entorno

Traigo el repositorio y me aseguro de que estén las tres bibliotecas de árboles. Colab ya
trae `xgboost` y `lightgbm`; `catboost` y `shap` casi nunca.
"""),
(CODE, r"""
import sys, subprocess, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
EN_COLAB = "google.colab" in sys.modules

if EN_COLAB:
    REPO = "https://github.com/debugsito/tech-salary-prediction.git"
    if not Path("tech-salary-prediction").exists():
        subprocess.run(["git", "clone", "--depth", "1", "-q", REPO], check=True)
    RAIZ = Path("/content/tech-salary-prediction")
    faltan = []
    for paquete, modulo in [("catboost", "catboost"), ("shap", "shap")]:
        try:
            __import__(modulo)
        except ImportError:
            faltan.append(paquete)
    if faltan:
        print("Instalando:", ", ".join(faltan))
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *faltan], check=True)
else:
    RAIZ = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()

sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "scripts"))
print("Raíz del proyecto:", RAIZ)
print("En Colab:", EN_COLAB)
"""),
(CODE, r"""
# Anoto las versiones antes de nada. El requirements.txt del repositorio fija las
# del Capítulo IV (Python 3.14, pandas 3.0.5) y Colab no las tiene: instalarlas
# rompería el entorno. Prefiero dejar constancia de con qué se corrió realmente.
import platform
import numpy as np, pandas as pd, sklearn, scipy
import xgboost, lightgbm, catboost, shap, matplotlib

print(f"{'Python':14} {platform.python_version():12}  (tesis: 3.14.7)")
for nombre, mod, esperada in [
    ("pandas", pd, "3.0.5"), ("numpy", np, "2.5.2"),
    ("scikit-learn", sklearn, "1.9.0"), ("scipy", scipy, "1.18.1"),
    ("xgboost", xgboost, "3.4.1"), ("lightgbm", lightgbm, "4.7.0"),
    ("catboost", catboost, "1.2.10"), ("shap", shap, "0.52.0"),
    ("matplotlib", matplotlib, "3.11.1"),
]:
    marca = "=" if mod.__version__ == esperada else "≠"
    print(f"{nombre:14} {mod.__version__:12} {marca} (tesis: {esperada})")
"""),
(MD, """
Ejecutado dentro del repositorio salen todas iguales, porque es el mismo entorno del
Capítulo IV. En Colab no van a coincidir casi ninguna: Colab trae su propio Python y sus
propias versiones, y forzar las del `requirements.txt` rompería la sesión.

Lo dejo anotado y no lo arreglo. Si más adelante alguna cifra no cuadra, este es el primer
sitio donde mirar, y además tiene interés por sí mismo: si los resultados aguantan un
cambio de versiones, es que no dependían de una en concreto.
"""),

# ---------------------------------------------------------------------------
(MD, """
## 1. Los datos

`cargar_encuesta` descarga la edición si no la encuentra en disco y aplica los seis
criterios de inclusión de la sección 3.2.4, devolviendo además el registro de cuántas
observaciones eliminó cada uno.
"""),
(CODE, r"""
import time
from src.carga_encuesta import cargar_encuesta, TARGET

t0 = time.perf_counter()
df, registro = cargar_encuesta(anio="2023")
print(f"\n{time.perf_counter() - t0:.0f} segundos\n")
print(registro.to_string(index=False))
"""),
(MD, """
Dos segundos, pero porque el archivo ya estaba en disco. La primera vez hay que bajar unos
150 MB del repositorio de Stack Overflow y la cosa cambia bastante.

De las 89,184 respuestas de la edición quedan 37,622, o sea que se cae el 58 %. Casi todo
el descarte está en el primer paso: 41,165 personas no declararon compensación. Los demás
filtros recortan mucho menos de lo que esperaba, y el de plausibilidad, que es el que más
discutimos al escribir el Capítulo III, solo se lleva 1,718.
"""),
(CODE, r"""
# La tesis da estas cifras en la tabla 4.2. Las comparo una a una en vez de
# mirarlas por encima, que es donde se cuelan los errores.
ESPERADO_TABLA_4_2 = [
    ("Respuestas totales de la edición",                         89184),
    ("Reporta compensación anual",                               48019),
    ("Situación de empleo: asalariado",                          43776),
    ("Desarrollador profesional",                                40410),
    ("País con al menos 30 observaciones",                       39682),
    ("Compensación dentro de [0.2, 6] veces la mediana del país", 37964),
    ("Coherencia entre años declarados y tramo de edad",          37622),
]
for criterio, esperado in ESPERADO_TABLA_4_2:
    fila = registro[registro["criterio"] == criterio]
    obtenido = int(fila["n"].iloc[0]) if len(fila) else None
    marca = "ok" if obtenido == esperado else "DISCREPA"
    print(f"{marca:9} {criterio:56} tesis {esperado:>7,}   ahora "
          f"{obtenido:>7,}" if obtenido is not None else f"{marca:9} {criterio:56} no encontrado")
"""),
(MD, """
Los siete pasos coinciden, hasta la última fila.

Esperaba una diferencia de una o dos observaciones en algún paso, porque el orden en que se
aplican los filtros puede alterar los recuentos intermedios. No la hay, y la razón es que el
módulo de carga fija el orden en el código en lugar de dejarlo al criterio de quien lo llama.
"""),
(CODE, r"""
import matplotlib.pyplot as plt

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.4))
a1.hist(df[TARGET] / 1000, bins=60, color="#4c72b0", edgecolor="white", linewidth=.3)
a1.set_xlabel("Compensación anual (miles de USD)")
a1.set_ylabel("Observaciones")
a1.set_title("Escala original")
a2.hist(df["salary_log"], bins=60, color="#55a868", edgecolor="white", linewidth=.3)
a2.set_xlabel("log(1 + compensación)")
a2.set_title("Escala logarítmica")
for a in (a1, a2):
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
plt.show()

from scipy.stats import skew
print(f"Asimetría en escala original:     {skew(df[TARGET]):.2f}")
print(f"Asimetría en escala logarítmica:  {skew(df['salary_log']):.2f}   (tesis: -0.82)")
"""),
(MD, """
La asimetría pasa de 2.56 a −0.82. Es lo que dice la tesis, aunque conviene leerlo con
cuidado: la transformación logarítmica **no deja la variable simétrica**, la pasa de muy
asimétrica por la derecha a algo asimétrica por la izquierda.

En el histograma de la derecha se ve el porqué: hay una cola de salarios bajos que en escala
logarítmica se estira. Aun así, −0.82 frente a 2.56 es una mejora grande, y es lo que
justifica modelar el logaritmo y no el salario directamente.
"""),
(CODE, r"""
comp = (df.groupby("income_group", observed=True)[TARGET]
          .agg(n="size", mediana="median", p25=lambda s: s.quantile(.25),
               p75=lambda s: s.quantile(.75)).round(0))
comp["% muestra"] = (comp["n"] / len(df) * 100).round(1)
print(comp.to_string())

# La dispersión hay que medirla DENTRO de cada país y luego resumirla por grupo.
# Si se calcula sobre el grupo entero se está midiendo la diferencia entre
# países, que es justo lo que no interesa aquí.
disp = (df.groupby("income_group", observed=True)
          .apply(lambda g: g.groupby("Country")["salary_log"].std().median(),
                 include_groups=False)
          .rename("desv. típica intra-país").to_frame())
disp["razón frente a renta alta"] = (disp["desv. típica intra-país"] /
                                     disp.loc["High income", "desv. típica intra-país"])
print()
print(disp.round(3).to_string())
"""),
(MD, """
El 85 % de la muestra son países de renta alta. Está muy desequilibrado, y es el reparto de
quien responde la encuesta, no el del mercado.

La segunda tabla es la que de verdad importa y tardé en entender por qué. Mide la desviación
típica del salario **dentro de cada país**, no dentro del grupo de renta: si se calcula
sobre el grupo entero se acaba midiendo la diferencia entre países, que es otra cosa. Medido
bien, sale 0.477 en renta alta frente a 0.781 en renta media-baja, un 64 % más.

Es decir: dos personas del mismo país de renta media-baja, con perfiles parecidos según las
variables que hay, cobran cosas mucho más distintas que dos personas del mismo país de renta
alta. Me lo apunto porque intuyo que va a hacer falta más adelante.
"""),

# ---------------------------------------------------------------------------
(MD, """
## 2. Diseño experimental

Dos estrategias de codificación para las variables de alta cardinalidad (país y rol)
contra siete modelos, todo sobre las mismas particiones para que la comparación admita
contraste pareado.
"""),
(CODE, r"""
import ejecutar_experimento as exp
from src.tuberia_caracteristicas import preparar_xy, construir_preprocesador, nombres_de_variables
from sklearn.model_selection import train_test_split

# Modo rápido: submuestra y dos repeticiones en lugar de cuatro. Sirve para ver
# que la cadena entera funciona; las cifras no son comparables con las del
# documento y el cuaderno lo avisa donde toca.
MODO_RAPIDO = False

X, y, A = preparar_xy(df)
estratos = df[exp.ESTRATO].astype(str)

if MODO_RAPIDO:
    exp.N_REPETICIONES = 2
    idx = X.sample(8000, random_state=exp.SEMILLA).index
    X, y, A, estratos = X.loc[idx], y[X.index.get_indexer(idx)], A.loc[idx], estratos.loc[idx]
    df_ref = df.loc[idx]
else:
    df_ref = df

X_ent, X_pru, y_ent, y_pru, A_ent, A_pru, e_ent, _ = train_test_split(
    X, y, A, estratos, test_size=exp.PROPORCION_PRUEBA,
    random_state=exp.SEMILLA, stratify=estratos)

print(f"Entrenamiento {len(X_ent):,}   prueba {len(X_pru):,}   columnas de entrada {X.shape[1]}")
print(f"Particiones: {exp.N_PARTICIONES} × {exp.N_REPETICIONES} repeticiones "
      f"= {exp.N_PARTICIONES * exp.N_REPETICIONES} pares para el contraste")

# Dos recuentos distintos que conviene no confundir: las columnas que entran al
# preprocesador y las que salen de él. La codificación por objetivo resume cada
# categórica de alta cardinalidad en una columna; la disyuntiva abre una por
# categoría, y de ahí la diferencia.
pre_t = construir_preprocesador(df_ref, "target").fit(X_ent, y_ent)
pre_o = construir_preprocesador(df_ref, "onehot").fit(X_ent, y_ent)
print(f"Variables de entrada:            {X.shape[1]}")
print(f"Columnas tras codificar (target): {len(nombres_de_variables(pre_t))}")
print(f"Columnas tras codificar (onehot): {len(nombres_de_variables(pre_o))}")
"""),
(MD, """
Tres recuentos que al principio confundí entre sí. Entran 119 variables al preprocesador;
salen 162 columnas si se codifica por objetivo y 261 si se codifica de forma disyuntiva. La
diferencia es toda de `Country` y `DevType`: la codificación por objetivo las resume en una
columna cada una y la disyuntiva abre una por categoría.

Lo de las cuatro repeticiones no es un capricho. Con cinco particiones el contraste de
Wilcoxon tiene un valor p mínimo alcanzable de 0.0625, que está por encima del 0.05, de modo
que ninguna hipótesis podría declararse significativa aunque la diferencia fuera enorme. Con
veinte pares el mínimo alcanzable baja a 8 × 10⁻⁶.
"""),

# ---------------------------------------------------------------------------
(MD, """
## 3. Entrenamiento

Aquí está el grueso del tiempo. Cada modelo se ajusta una vez por partición y por
estrategia de codificación, y el preprocesador se reajusta dentro de cada partición
porque la codificación por objetivo usa la variable dependiente y ajustarla fuera
filtraría información del conjunto de validación.
"""),
(CODE, r"""
modelos = exp.construir_configuraciones()
print("Modelos:", ", ".join(modelos))

t0 = time.perf_counter()
cv = pd.concat([
    exp.evaluar(X_ent, y_ent, e_ent, modelos, codificacion, df_ref)
    for codificacion in ("target", "onehot")
], ignore_index=True)
print(f"\n{time.perf_counter() - t0:.0f} segundos, {len(cv)} ajustes")
"""),
(MD, """
235 segundos para 280 ajustes. En Colab habrá que contar bastante más, porque la CPU que da
es más lenta y los modelos de árboles se benefician mucho de los núcleos.

Siete modelos por veinte particiones por dos codificaciones son 280, así que no falló
ninguno. Me esperaba algún aviso de CatBoost, que es el más quisquilloso con las categóricas,
y no salió ninguno.
"""),
(CODE, r"""
resumen = (cv.groupby(["modelo", "codificacion"])
             .agg(R2=("R2", "mean"), R2_de=("R2", "std"),
                  MAE_USD=("MAE_USD", "mean"), seg=("segundos", "mean"))
             .round(4).sort_values("R2", ascending=False))
print(resumen.to_string())
"""),
(MD, """
XGBoost con codificación por objetivo queda primero, con R² de 0.7863. La tesis da esa misma
cifra.

Lo que no me esperaba es lo apretado que está todo arriba: entre el primero y el quinto hay
tres milésimas de R², y la desviación típica entre particiones es de seis milésimas. O sea
que **la diferencia entre los cuatro ensamblados es menor que la variabilidad entre
particiones de cualquiera de ellos**. Elegir uno u otro por su media sin mirar la dispersión
sería quedarse con ruido.

Ridge se queda en 0.7544, tres puntos por debajo. Las líneas base dan R² de cero o negativo,
que es lo que tienen que dar: no usan ninguna variable.

La columna de segundos también dice algo. CatBoost tarda cuatro veces más que XGBoost para
el mismo resultado.
"""),
(CODE, r"""
# Lo mismo sobre el conjunto de prueba, que no ha intervenido en nada hasta ahora.
from sklearn.pipeline import Pipeline
from src.metricas import calcular_metricas

filas = []
tuberias = {}
for codificacion in ("target", "onehot"):
    for nombre, modelo in modelos.items():
        from sklearn.base import clone
        t = Pipeline([("preprocesador", construir_preprocesador(df_ref, codificacion)),
                      ("modelo", clone(modelo))])
        t.fit(X_ent, y_ent)
        m = calcular_metricas(y_pru, t.predict(X_pru))
        m.update({"modelo": nombre, "codificacion": codificacion})
        filas.append(m)
        tuberias[(nombre, codificacion)] = t

prueba = pd.DataFrame(filas).set_index(["modelo", "codificacion"])
print(prueba[["R2", "MAE_USD", "MedAE_USD", "MAPE"]].round(4).sort_values("R2", ascending=False).to_string())
"""),
(MD, """
Sobre el conjunto de prueba el orden cambia: se pone delante CatBoost con 0.7865 y XGBoost
baja al segundo puesto por tres diezmilésimas.

No lo leo como que CatBoost sea mejor. Lo leo como confirmación de lo de antes: las
diferencias entre estos modelos están por debajo del ruido, y en un conjunto de prueba de
7,525 observaciones ese ruido basta para reordenarlos. Si hubiera que elegir uno para
producción, yo miraría el tiempo de ajuste antes que la cuarta cifra decimal.

Lo importante es que las cifras de prueba se parecen a las de validación cruzada. Si el
modelo estuviera sobreajustado, aquí se habría notado.
"""),

# ---------------------------------------------------------------------------
(MD, """
## 4. Contraste de hipótesis

Wilcoxon pareada sobre las particiones, con corrección de Holm-Bonferroni dentro de cada
familia de comparaciones.
"""),
(CODE, r"""
contrastes = exp.contrastes(cv, metrica="R2")
if contrastes.empty:
    raise SystemExit("Sin contrastes: hacen falta al menos seis particiones pareadas.")

for familia in contrastes["familia"].unique():
    print(f"\n{familia}")
    bloque = contrastes[contrastes["familia"] == familia]
    print(bloque[["config_a", "config_b", "n_pares", "media_a", "media_b",
                  "diferencia", "p_valor", "significativo"]]
          .to_string(index=False))
"""),
(MD, """
Tres cosas que leo de aquí.

**La hipótesis general se sostiene.** Los veinte contrastes contra las líneas base salen
significativos con p de 2 × 10⁻⁶, que es el mínimo alcanzable con veinte pares. Era
esperable: la diferencia entre R² de 0.78 y R² de 0.00 no necesita una prueba estadística
para verse.

**HE1 solo se sostiene a medias, y esto sí es interesante.** La codificación por objetivo
gana de forma significativa en XGBoost y CatBoost, pero en Ridge **pierde** de forma
significativa, y en LightGBM y HistGradientBoosting el resultado no es significativo. Así
que la respuesta no es «la codificación por objetivo es mejor», sino «depende del modelo».
Con Ridge es peor, y tiene sentido: un modelo lineal no puede aprovechar una variable que
resume la categoría en un solo número, mientras que un árbol sí puede partirla.

**Las magnitudes son ridículas.** La mayor diferencia entre codificaciones es de 0.0040 de
R². Es significativa porque se repite en las veinte particiones, no porque sea grande.
Significación y relevancia no son lo mismo, y aquí se ve de manual.
"""),

# ---------------------------------------------------------------------------
(MD, """
## 5. Explicabilidad

SHAP sobre el mejor modelo, para ver de qué depende realmente la estimación.
"""),
(CODE, r"""
from src.explicabilidad import explicar

mejor = resumen.index[0]
print("Modelo explicado:", mejor)

t0 = time.perf_counter()
shap_res = explicar(tuberias[mejor], X_ent, X_pru,
                    nombres=nombres_de_variables(tuberias[mejor].named_steps["preprocesador"]),
                    directorio=str(RAIZ / "resultados_colab"),
                    etiqueta=f"{mejor[0]}_{mejor[1]}_2023",
                    n_explicar=4000 if MODO_RAPIDO else 20000)
print(f"{time.perf_counter() - t0:.0f} segundos")
print("Explicador:", shap_res["estimador"], "·", shap_res["n_variables"], "variables")
sp = shap_res.get("spearman_shap_vs_impureza")
if sp:
    print(f"Spearman SHAP frente a importancia por impureza: rho = {sp['rho']}, "
          f"p = {sp['p_valor']} sobre {sp['n_variables']} variables")

# La función deja el resumen en un CSV; lo leo de ahí en vez de recalcularlo.
imp = pd.read_csv(shap_res["ruta_resumen"]).set_index("variable")["shap_medio_abs"]
top = imp.sort_values(ascending=False).head(15)

fig, ax = plt.subplots(figsize=(7.5, 4.6))
ax.barh(range(len(top)), top.values[::-1], color="#4c72b0")
ax.set_yticks(range(len(top)), [n[:38] for n in top.index[::-1]], fontsize=8)
ax.set_xlabel("Contribución media absoluta (escala log)")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
plt.show()
"""),
(MD, """
Un segundo. Me quedé mirando la pantalla esperando que tardara y ya había terminado: al ser
XGBoost un modelo de árboles, SHAP usa `TreeExplainer`, que es exacto y rápido, en lugar de
aproximar por muestreo.

El país domina con diferencia, y a bastante distancia de lo segundo. La correlación de
Spearman entre el orden que da SHAP y el que da la importancia por impureza es 0.75: alta,
pero lejos de 1. Son dos cosas distintas aunque las llamen igual, y un cuarto de la
ordenación cambia según cuál se use.
"""),
(CODE, r"""
# La tesis agrupa las variables en bloques teóricos y compara su peso. Repito el
# agrupamiento con lo que acaba de salir.
def bloque(v):
    if v.startswith(("language__", "database__", "platform__", "webframe__")):
        return "Tecnologías"
    if v.startswith(("Country", "income_group", "wb_region")):
        return "Geográfico"
    if v.startswith(("YearsCode", "WorkExp", "EdLevel", "DevType", "ICorPM")):
        return "Capital humano"
    if v.startswith(("Age", "Gender")):
        return "Demográfico"
    return "Organizacional"

por_bloque = (imp.rename("shap")
                .to_frame().assign(bloque=lambda d: d.index.map(bloque))
                .groupby("bloque")["shap"].agg(["sum", "size"])
                .sort_values("sum", ascending=False))
por_bloque["% del total"] = (por_bloque["sum"] / por_bloque["sum"].sum() * 100).round(1)
print(por_bloque.round(4).to_string())
"""),
(MD, """
El bloque geográfico se lleva el 35 % de la contribución total con solo cuatro variables,
mientras que el de tecnologías necesita 106 para llegar al 26 %.

El demográfico se queda en el 1.5 % con siete variables. Eso quiere decir que la edad y las
variables de ese bloque casi no intervienen en la estimación, cosa que conviene tener a mano
cuando se hable de equidad.
"""),

# ---------------------------------------------------------------------------
(MD, """
## 6. Auditoría de equidad

El modelo se evalúa por grupos, no en agregado. La tesis insiste en el error **relativo**
a la mediana del grupo, y no en el absoluto, porque comparar errores en dólares entre
países con medianas que difieren en un orden de magnitud no dice nada.
"""),
(CODE, r"""
from src.equidad import auditar, resumir

pred = tuberias[mejor].predict(X_pru)
aud = auditar(y_pru, pred, A_pru["income_group"])
print(resumir(aud, "Nivel de renta del país"))
"""),
(MD, """
Aquí está lo que a mí me parece el resultado más interesante de todo el trabajo, y tuve que
leerlo dos veces para verlo.

Mirando el MAE en dólares, el grupo peor servido sería el de renta alta: 26,286 dólares de
error frente a 9,846 en renta media-baja. Parecería que el modelo funciona mucho peor con
los países ricos.

Dividiendo cada error por la mediana de su propio grupo, **la conclusión se da la vuelta**:
29.8 % en renta alta frente a 58.0 % en renta media-baja. El modelo se equivoca en dólares
más con los países ricos simplemente porque allí los salarios son más grandes; en proporción
al salario típico de cada sitio, se equivoca el doble con los pobres.

El propio resumen lo avisa con una línea que marca que el grupo peor servido cambia al
normalizar. Si la auditoría se hubiera hecho con el error absoluto, la conclusión publicada
habría sido la contraria a la correcta.
"""),
(CODE, r"""
g = pd.DataFrame(aud["por_grupo"]).T
g = g[["n", "mae", "mae_relativo", "sesgo_sistematico"]].astype(float).sort_values("mae_relativo")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.4))
a1.barh(g.index, g["mae"] / 1000, color="#c44e52")
a1.set_xlabel("MAE (miles de USD)"); a1.set_title("Error absoluto")
a2.barh(g.index, g["mae_relativo"], color="#4c72b0")
a2.set_xlabel("MAE / mediana del grupo"); a2.set_title("Error relativo")
for a in (a1, a2):
    a.spines[["top", "right"]].set_visible(False)
    a.tick_params(labelsize=8)
fig.tight_layout()
plt.show()

print(f"Razón de disparidad con error absoluto: {g['mae'].max() / g['mae'].min():.2f}")
print(f"Razón de disparidad con error relativo: {g['mae_relativo'].max() / g['mae_relativo'].min():.2f}")
"""),
(MD, """
Los dos gráficos, uno al lado del otro, dicen lo contrario el uno del otro. Es el mismo
modelo, el mismo conjunto de prueba y los mismos grupos: lo único que cambia es si el error
se divide por la escala del grupo.

La razón de disparidad pasa de 2.67 a 1.95. Las dos superan el umbral de 1.25 que fija el
Capítulo II, así que la disparidad existe de cualquier modo; lo que cambia es **a quién
perjudica**.
"""),

# ---------------------------------------------------------------------------
(MD, """
## 7. El género, que solo está en la edición 2022

Las ediciones 2023 y 2025 retiraron las variables demográficas protegidas. Para auditar
género hay que bajar la edición 2022 y repetir el ajuste sobre ella.
"""),
(CODE, r"""
df22, reg22 = cargar_encuesta(anio="2022")
print(f"Edición 2022: {len(df22):,} observaciones efectivas")
declarado = (df22["Gender"] != "No declarado").sum()
print(f"Con género declarado: {declarado:,}  ·  sin declarar: {len(df22) - declarado:,}")

X22, y22, A22 = preparar_xy(df22)
e22 = df22[exp.ESTRATO].astype(str)
Xe, Xp, ye, yp, Ae, Ap = train_test_split(
    X22, y22, A22, test_size=exp.PROPORCION_PRUEBA,
    random_state=exp.SEMILLA, stratify=e22)

from sklearn.base import clone
t22 = Pipeline([("preprocesador", construir_preprocesador(df22, "target")),
                ("modelo", clone(modelos[mejor[0]]))])
t22.fit(Xe, ye)
p22 = t22.predict(Xp)
print(f"R² sobre prueba (2022): {calcular_metricas(yp, p22)['R2']:.4f}")

aud_g = auditar(yp, p22, Ap["Gender"])
print()
print(resumir(aud_g, "Género"))
"""),
(MD, """
De 28,951 observaciones efectivas, 28,609 declaran género. Hay que usar la edición 2022
porque las de 2023 y 2025 retiraron las variables demográficas, así que esta parte no se
puede repetir con datos más recientes.

El R² sobre 2022 es 0.7799, muy parecido al de 2023. Al menos el modelo se comporta igual en
las dos ediciones.

La tabla sale con once grupos porque la encuesta permite marcar varias opciones a la vez, y
la mayoría son combinaciones con dos o tres personas. La función las marca como
descriptivas y las deja fuera de la inferencia, que es lo correcto, pero aun así ensucian
bastante la lectura. Me quedo con los cuatro que superan las treinta observaciones.
"""),
(CODE, r"""
# Lo que de verdad quiero saber: ¿el modelo estima peor a las mujeres, o estima
# igual de bien un mercado que ya paga distinto? Separo las dos cosas.
gen = pd.DataFrame(aud_g["por_grupo"]).T
gen = gen[gen["n"].astype(int) >= 30]
salario = (pd.DataFrame({"g": Ap["Gender"].values, "y": np.expm1(yp)})
             .dropna().groupby("g")["y"].median())

tabla = pd.DataFrame({
    "n": gen["n"].astype(int),
    "mediana real (USD)": salario.reindex(gen.index).round(0),
    "MAE relativo": gen["mae_relativo"].astype(float).round(3),
    "sesgo (USD)": gen["sesgo_sistematico"].astype(float).round(0),
})
print(tabla.to_string())
"""),
(MD, """
Esto es lo que quería ver. Los hombres tienen un error relativo de 0.360 y las mujeres de
0.362: dos milésimas de diferencia, sobre 5,310 y 286 observaciones respectivamente.

Y a la vez la mediana real de las mujeres está por debajo de la de los hombres. **La brecha
está en los datos, pero no en el error del modelo.** El sistema estima igual de bien a unas
y a otros; lo que ocurre es que el mercado que describe paga distinto.

Son dos cosas que es fácil confundir y que llevan a conclusiones opuestas. Un modelo puede
reproducir fielmente un mercado desigual sin ser él mismo desigual, y ahí la responsabilidad
está en usarlo o no, no en su exactitud.

Sobre la razón de disparidad de 1.347 que sale arriba: no viene del contraste entre hombres
y mujeres, sino del grupo de quienes no declararon género, que son setenta personas y tienen
un error relativo de 0.486. Conviene decirlo así y no dejar el 1.347 suelto, porque leído sin
contexto sugiere una brecha de género que los datos no muestran.
"""),

# ---------------------------------------------------------------------------
(MD, """
## 8. Contraste con lo que afirma la tesis

Esto es lo que me interesaba de verdad al montar el cuaderno.
"""),
(CODE, r"""
# Tolerancias: el R² se compara con margen porque el orden de las operaciones en
# coma flotante y las versiones de las bibliotecas lo mueven en los decimales
# altos. Las cifras de referencia son las de los artefactos del repositorio, que
# es de donde el documento toma las suyas.
razon_rel = g["mae_relativo"].max() / g["mae_relativo"].min()
AFIRMA_LA_TESIS = [
    ("Muestra efectiva 2023",             len(df),                      37622,  0),
    ("Variables de entrada",              X.shape[1],                   119,    0),
    ("Columnas tras codificar (target)",  len(nombres_de_variables(pre_t)), 162, 0),
    ("Pares para el contraste",           exp.N_PARTICIONES * exp.N_REPETICIONES, 20, 0),
    ("R² en validación, mejor config.",   resumen["R2"].max(),          0.7863, 0.010),
    ("R² en prueba, mejor config.",       prueba["R2"].max(),           0.7865, 0.010),
    ("R² de Ridge en validación (onehot)", resumen.loc[("Ridge", "onehot"), "R2"], 0.7544, 0.010),
    ("MAE relativo, renta media-baja",    float(g.loc["Lower middle income", "mae_relativo"]), 0.580, 0.05),
    ("MAE relativo, renta alta",          float(g.loc["High income", "mae_relativo"]),         0.298, 0.05),
    ("Razón de disparidad relativa",      razon_rel,                    1.947,  0.15),
    ("Razón de disparidad absoluta",      g["mae"].max() / g["mae"].min(), 2.670, 0.25),
]

print(f"{'':9} {'Magnitud':36} {'ahora':>10} {'tesis':>10}")
fallos = []
for nombre, obtenido, esperado, tol in AFIRMA_LA_TESIS:
    ok = abs(float(obtenido) - esperado) <= tol
    if not ok:
        fallos.append(nombre)
    fmt = ",.0f" if esperado >= 100 else ".4f"
    print(f"{'ok' if ok else 'DISCREPA':9} {nombre:36} {obtenido:>10{fmt}} {esperado:>10{fmt}}")

sig = contrastes[contrastes["familia"] == "HG_vs_baseline"]["significativo"]
print(f"\n{'ok' if sig.all() else 'DISCREPA':9} "
      f"Hipótesis general: los {len(sig)} contrastes contra la línea base salen significativos")

he1 = contrastes[contrastes["familia"] == "HE1_codificacion"]
print(f"{'':9} HE1: {int(he1['significativo'].sum())} de {len(he1)} comparaciones "
      f"de codificación resultan significativas")

if MODO_RAPIDO:
    print("\nEl modo rápido usa submuestra y menos repeticiones: las discrepancias "
          "de arriba son esperables y no dicen nada sobre el documento.")
elif not fallos:
    print("\nTodo dentro de tolerancia.")
else:
    print("\nRevisar:", ", ".join(fallos))
"""),
(MD, """
Las once magnitudes caen dentro de tolerancia y los veinte contrastes de la hipótesis general
salen significativos.

Sobre HE1 el propio contraste ya avisaba: tres de cinco comparaciones son significativas, no
las cinco. El documento lo recoge así.

Queda dicho lo que este cuaderno **no** prueba. Que las cifras se reproduzcan significa que
la cadena de análisis es determinista y que el documento no inventó números; no significa que
las decisiones de diseño sean las acertadas. Los criterios de inclusión, el umbral de
plausibilidad relativo al país y la elección del error relativo como indicador de referencia
son decisiones argumentadas en los Capítulos III y V, y se discuten allí, no aquí.
"""),
]


def main():
    nb = {
        "cells": [celda(t, f) for t, f in CELDAS],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python"},
            "colab": {"provenance": [], "toc_visible": True},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    DESTINO.parent.mkdir(exist_ok=True)
    DESTINO.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    n_code = sum(1 for t, _ in CELDAS if t == CODE)
    print(f"{DESTINO.relative_to(RAIZ)}: {len(CELDAS)} celdas ({n_code} de código)")


if __name__ == "__main__":
    main()
