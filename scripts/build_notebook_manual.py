#!/usr/bin/env python3
"""Construye el cuaderno de trabajo, el que recorre el análisis paso a paso.

Hay tres cuadernos y conviene no confundirlos:

    tesis_reproducible.ipynb  Recorre los artefactos ya calculados. Rápido.
    tesis_colab.ipynb         Reejecuta el experimento y lo contrasta con el
                              documento. Sirve de verificación.
    tesis_trabajo.ipynb       Este. Rehace el análisis entero desde el CSV en
                              bruto, decisión por decisión, y produce todas las
                              figuras y tablas del documento.

El tercero no llama a los scripts de análisis: reconstruye lo que hacen, de modo
que cada filtro, cada partición y cada métrica estén a la vista. Sí reutiliza las
funciones de conversión del módulo de carga (fechas, regiones, columnas
multivalor), que no encierran decisión alguna, y las funciones de figuras, para
que las imágenes salgan idénticas a las del documento.

Se genera y no se edita a mano. Para cambiar algo, se edita este archivo.

Uso:
    python scripts/build_notebook_manual.py
    jupyter nbconvert --execute --to notebook --inplace notebooks/tesis_trabajo.ipynb
"""

import json
from pathlib import Path

MD = "markdown"
CODE = "code"
RAIZ = Path(__file__).resolve().parent.parent
DESTINO = RAIZ / "notebooks" / "tesis_trabajo.ipynb"


def celda(tipo, fuente, n=[0]):
    n[0] += 1
    base = {"cell_type": tipo, "id": f"c{n[0]:02d}", "metadata": {},
            "source": fuente.strip().split("\n")}
    base["source"] = [l + "\n" for l in base["source"][:-1]] + [base["source"][-1]]
    if tipo == CODE:
        base.update({"execution_count": None, "outputs": []})
    return base


CELDAS = [

(MD, """
# Predicción de salarios en el sector tecnológico

Cuaderno de trabajo. Va desde el archivo de la encuesta hasta las conclusiones,
sin saltarse pasos.

**Para ejecutarlo en tu equipo**

```
git clone https://github.com/debugsito/tech-salary-prediction.git
cd tech-salary-prediction
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/jupyter lab notebooks/tesis_trabajo.ipynb
```

**Para ejecutarlo en Colab**: abre el enlace y pulsa «Ejecutar todo». La primera
celda se encarga del resto.

https://colab.research.google.com/github/debugsito/tech-salary-prediction/blob/main/notebooks/tesis_trabajo.ipynb

Descarga unos 400 MB de datos y tarda entre quince y cuarenta minutos según la
máquina. La semilla está fija en 42 en todas partes, así que dos ejecuciones dan
lo mismo.
"""),

(CODE, r"""
import sys, subprocess
from pathlib import Path

if "google.colab" in sys.modules:
    if not Path("tech-salary-prediction").exists():
        subprocess.run(["git", "clone", "--depth", "1", "-q",
                        "https://github.com/debugsito/tech-salary-prediction.git"], check=True)
    RAIZ = Path("/content/tech-salary-prediction")
    for paquete in ("catboost", "shap"):
        try:
            __import__(paquete)
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", paquete], check=True)
else:
    RAIZ = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()

sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "scripts"))

import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display

def figura(ruta):
    # Muestra la imagen, o avisa si faltaba el artefacto del que sale.
    if ruta is None:
        print("No se ha podido generar: falta el artefacto de origen.")
        return
    display(Image(ruta))

SEMILLA = 42

# Lo que produzca este cuaderno va a su propio directorio y no encima de
# results/ y figures/, que son los artefactos del proyecto. Si una celda falla a
# mitad, prefiero que no se lleve por delante nada que ya estaba bien.
SALIDA = RAIZ / "salida_cuaderno" / "results"
FIGURAS = RAIZ / "salida_cuaderno" / "figures"
SALIDA.mkdir(parents=True, exist_ok=True)
FIGURAS.mkdir(parents=True, exist_ok=True)

pd.set_option("display.width", 120)
print("Listo. Raíz:", RAIZ)
"""),

# ---------------------------------------------------------------------------
(MD, """
## 1. Traer los datos y ver qué hay

La encuesta de Stack Overflow se publica en su propio repositorio, un CSV por
edición. Empiezo por la de 2023.
"""),
(CODE, r"""
from src.data_loader_stackoverflow import descargar, COLUMNAS, TARGET

ruta = descargar("2023")
print(ruta, f"{ruta.stat().st_size / 1e6:.0f} MB")

# El archivo trae unas ochenta columnas y no necesito casi ninguna. Leo primero
# la cabecera para quedarme con las que sí.
cabecera = pd.read_csv(ruta, nrows=0)
print(f"\nColumnas en el archivo: {len(cabecera.columns)}")
usadas = [c for c in COLUMNAS if c in cabecera.columns]
print(f"Columnas que me quedo:   {len(usadas)}")
print(", ".join(usadas))
"""),
(MD, """
84 columnas en el archivo y me quedo con 21. La encuesta pregunta muchas cosas que
no tienen que ver con el salario (qué editor usas, qué sistema operativo) y
arrastrarlas solo haría la tabla más pesada.
"""),
(CODE, r"""
bruto = pd.read_csv(ruta, usecols=usadas, low_memory=False)
print(f"{len(bruto):,} respuestas × {bruto.shape[1]} columnas\n")
bruto.head(3).T
"""),
(MD, """
89,184 respuestas. Mirando las dos primeras ya se ve algo: la primera persona
tiene casi todo vacío porque en `MainBranch` marcó «None of these», así que
probablemente abandonó el cuestionario al principio.

También veo que las tecnologías vienen como una sola cadena con punto y coma
(`HTML/CSS;JavaScript;Python`). Eso habrá que separarlo.
"""),

# ---------------------------------------------------------------------------
(MD, """
## 2. La variable que quiero predecir

`ConvertedCompYearly` es la compensación anual que la propia encuesta convierte a
dólares. Antes de nada, a ver qué pinta tiene.
"""),
(CODE, r"""
c = bruto[TARGET]
print(f"Declarada por {c.notna().sum():,} de {len(c):,} personas "
      f"({c.notna().mean()*100:.0f} %)\n")
print(c.describe().apply(lambda v: f"{v:,.0f}").to_string())
print(f"\nAsimetría: {c.skew():.2f}")
"""),
(MD, """
Solo el 54 % declara compensación. Es mucha pérdida, pero sin salario no hay nada
que predecir.

Y la variable está sucia de verdad. El mínimo es **un dólar** y el máximo **74
millones**. La asimetría de 94.7 no es de una distribución con cola larga, es de
una distribución con basura dentro. La desviación típica (681,419) es casi siete
veces la media (103,110), que es otra señal de lo mismo.
"""),
(CODE, r"""
# Miro los extremos para hacerme una idea de qué son esos valores.
extremos = bruto.loc[c.notna(), [TARGET, "Country", "Employment", "MainBranch"]]
print("Los cinco más altos:")
print(extremos.nlargest(5, TARGET).to_string(index=False))
print("\nLos cinco más bajos:")
print(extremos.nsmallest(5, TARGET).to_string(index=False))
"""),
(MD, """
Confirmado, y ayuda ver de qué tipo son los errores.

Los cinco más altos van de 36 a 74 millones de dólares. No es que sean
ejecutivos bien pagados: son cifras imposibles. Y llama la atención que tres de
los cinco sean autónomos, que quizá declararon facturación en vez de salario, o
la cantidad en su moneda local sin convertir.

Los cinco más bajos son todos de un dólar exacto, en India, Uzbekistán, Vietnam,
Irán. Un dólar exacto no es un salario bajo, es alguien que puso «1» para pasar
de la pregunta.

Me quedo con una idea para después: los errores parecen ser de **escala**
(declarar el sueldo mensual, o poner mal el separador decimal), no valores
extremos legítimos.
"""),

# ---------------------------------------------------------------------------
(MD, """
## 3. Limpiar, un criterio cada vez

Voy a ir aplicando filtros de uno en uno y anotando cuánto se lleva cada uno. Al
final quiero poder decir exactamente por qué cada persona que queda está dentro.
"""),
(CODE, r"""
from src.data_loader_stackoverflow import normalizar_ausentes

# La encuesta escribe la no respuesta como texto ("NA", "Prefer not to say"),
# que pandas no reconoce como nulo. Si no lo arreglo aquí, esas cadenas acabarán
# siendo categorías con nombre propio.
df = normalizar_ausentes(bruto.copy(), [c for c in usadas if c != TARGET])

registro = [("Respuestas totales de la edición", len(df))]
def anotar(criterio):
    descartados = registro[-1][1] - len(df)
    registro.append((criterio, len(df)))
    print(f"{criterio:58} quedan {len(df):>7,}   (−{descartados:,})")

print(f"{'Punto de partida':58} quedan {len(df):>7,}")
"""),
(CODE, r"""
# 1. Sin compensación declarada no hay nada que predecir.
df = df[df[TARGET].notna() & (df[TARGET] > 0)]
anotar("Reporta compensación anual")
"""),
(CODE, r"""
# 2. Solo asalariados. Los autónomos y quienes buscan empleo declaran ingresos
# que no son comparables con un salario.
df = df[df["Employment"].astype("string").str.startswith("Employed", na=False)]
anotar("Situación de empleo: asalariado")
"""),
(CODE, r"""
# 3. Solo quien se declara desarrollador de profesión. La encuesta la responde
# también mucha gente que programa por afición o estudia.
df = df[df["MainBranch"] == "I am a developer by profession"]
anotar("Desarrollador profesional")
"""),
(CODE, r"""
# 4. Países con pocas respuestas fuera. Con cinco o seis observaciones no hay
# forma de estimar nada del país, y además me hace falta su mediana en el paso
# siguiente.
frecuencia = df["Country"].value_counts()
print(f"Países antes: {len(frecuencia)}")
print(f"  con menos de 30 respuestas: {(frecuencia < 30).sum()}")
df = df[df["Country"].isin(frecuencia[frecuencia >= 30].index)]
anotar("País con al menos 30 observaciones")
print(f"Países después: {df['Country'].nunique()}")
"""),
(MD, """
El primer filtro se lleva 41,165 respuestas de golpe, el 46 % del total. Todos
los demás juntos no llegan ni de lejos.

Lo de los países me ha sorprendido: **87 de los 165 países tienen menos de 30
respuestas**, y entre los 87 solo suman 728 personas. Se van más de la mitad de
los países y menos del 2 % de la muestra.
"""),

# ---------------------------------------------------------------------------
(MD, """
### El filtro difícil: qué salario es creíble

Los extremos de antes no son salarios reales. Pero decidir dónde cortar no es
obvio, así que pruebo dos criterios y comparo.
"""),
(CODE, r"""
from scipy.stats import skew

antes = df.copy()

# Opción A: recortar por percentiles globales, que es lo primero que se me ocurrió.
p1, p99 = antes[TARGET].quantile([0.01, 0.99])
a = antes[(antes[TARGET] >= p1) & (antes[TARGET] <= p99)]

# Opción B: relativo a la mediana del propio país.
mediana_pais = antes.groupby("Country")[TARGET].transform("median")
b = antes[(antes[TARGET] >= mediana_pais * 0.2) & (antes[TARGET] <= mediana_pais * 6)]

print(f"Percentiles globales: corte en ${p1:,.0f} y ${p99:,.0f}")
print(f"{'':22} {'quedan':>8} {'%':>7} {'asimetría':>10} {'tras log':>9}")
for nombre, sub in [("Sin filtrar", antes), ("A) p1-p99 global", a), ("B) [0.2, 6] × país", b)]:
    print(f"{nombre:22} {len(sub):>8,} {len(sub)/len(antes)*100:>6.1f}% "
          f"{skew(sub[TARGET]):>10.2f} {skew(np.log1p(sub[TARGET])):>9.2f}")
"""),
(MD, """
Aquí está lo interesante, y no es lo que esperaba.

La opción A conserva más gente (98.1 % contra 95.7 %) y deja la variable sin
transformar mucho más ordenada (asimetría 1.37 contra 2.56). Si me quedara con
esas dos columnas, elegiría A sin dudarlo.

Pero la columna que importa es la última, porque el modelo no se va a ajustar
sobre el salario sino sobre su logaritmo. Y ahí **A deja −1.43 y B deja −0.82**,
casi la mitad. B conserva salarios altos que el percentil 99 tiraba, y por eso su
asimetría sin transformar sube; en escala logarítmica, que es donde se trabaja,
el resultado es bastante mejor.

Me falta ver una cosa antes de decidir: a quién le quita las observaciones cada
uno.
"""),
(CODE, r"""
# Cuál de los dos recortes cae sobre qué países, y por qué lado. Los recuentos
# en bruto engañan, porque los grupos son de tamaños muy distintos; miro el
# porcentaje de cada grupo que se lleva cada corte.
from src.data_loader_stackoverflow import derivar_region, cargar_referencia_paises
ref = cargar_referencia_paises()
con_renta = derivar_region(antes.copy(), ref)
med = con_renta.groupby("Country")[TARGET].transform("median")

print(f"{'':22} {'N':>7}   {'A: por abajo':>12} {'A: por arriba':>13}   "
      f"{'B: por abajo':>12} {'B: por arriba':>13}")
for g, sub in con_renta.groupby("income_group", observed=True):
    m, n = med.loc[sub.index], len(sub)
    print(f"{g:22} {n:>7,}   "
          f"{100*(sub[TARGET] < p1).sum()/n:>11.2f}% {100*(sub[TARGET] > p99).sum()/n:>12.2f}%   "
          f"{100*(sub[TARGET] < m*0.2).sum()/n:>11.2f}% {100*(sub[TARGET] > m*6).sum()/n:>12.2f}%")
"""),
(MD, """
Esto es lo que quería mirar, y el resultado tiene matiz.

**El corte de A cae casi todo por abajo**, en el umbral de 670 dólares: se lleva
el 4.20 % de los países de renta media-baja y solo el 0.48 % de los de renta
alta, casi nueve veces más. Un sueldo de 600 dólares al año es raro en Alemania
pero no es imposible en un país de renta baja, así que ahí A está tirando
observaciones que pueden ser buenas.

**B quita más en total**, eso es cierto: 11-13 % en los grupos de menor renta
frente al 1.94 % en renta alta. Pero la desproporción es menor (unas seis veces,
no nueve) y sobre todo está mejor repartida entre las dos colas. En los países de
renta media-baja B se lleva un 3.74 % por arriba, que son justo esas cifras
infladas por factor 10 o 12 de las que hablaba antes.

No voy a fingir que B sea inocuo con los países pobres, porque no lo es. Lo que
sí hace es cortar donde están los errores en vez de cortar por una cantidad fija
que significa cosas distintas en cada sitio. Con eso y con la asimetría de −0.82,
me quedo con B.
"""),
(CODE, r"""
df = b
anotar("Compensación dentro de [0.2, 6] veces la mediana del país")
"""),

# ---------------------------------------------------------------------------
(MD, """
### Último filtro: respuestas que se contradicen
"""),
(CODE, r"""
from src.data_loader_stackoverflow import parsear_anios, TRAMO_EDAD_MAX

anios_total = parsear_anios(df["YearsCode"])
anios_pro = parsear_anios(df["YearsCodePro"])

# Quien dice llevar más años programando profesionalmente que programando en
# total ha rellenado mal el formulario.
mas_pro_que_total = (anios_pro > anios_total).fillna(False)

# Y quien, restando los años que dice llevar programando al extremo superior de
# su tramo de edad, habría empezado antes de los ocho años.
edad_max = df["Age"].map(TRAMO_EDAD_MAX)
empezo_demasiado_pronto = ((edad_max - anios_total) < 8).fillna(False)

print(f"Más años profesionales que totales: {mas_pro_que_total.sum():,}")
print(f"Habría empezado antes de los 8 años: {empezo_demasiado_pronto.sum():,}")
print(f"Alguna de las dos:                  {(mas_pro_que_total | empezo_demasiado_pronto).sum():,}")

df = df[~(mas_pro_que_total | empezo_demasiado_pronto)].reset_index(drop=True)
anotar("Coherencia entre años declarados y tramo de edad")
"""),
(MD, """
342 respuestas se contradicen a sí mismas. La mayoría (287) dicen llevar más años
programando profesionalmente que programando en total, que es imposible.

Son pocas y da algo de pereza el filtro, pero son errores de captura, no valores
raros: no hay forma de interpretarlas literalmente, y si las dejo el modelo
aprende de ellas.
"""),
(CODE, r"""
tabla_filtros = pd.DataFrame(registro, columns=["Criterio", "N resultante"])
tabla_filtros["Descartados"] = tabla_filtros["N resultante"].diff().fillna(0).abs().astype(int)
tabla_filtros.loc[0, "Descartados"] = 0
print(tabla_filtros.to_string(index=False))
print(f"\nMe quedo con {len(df):,} de {registro[0][1]:,} "
      f"({len(df)/registro[0][1]*100:.1f} %) en {df['Country'].nunique()} países.")
"""),

# ---------------------------------------------------------------------------
(MD, """
## 4. Terminar de preparar la tabla

Faltan las variables derivadas: el logaritmo del salario, los años como número,
la región y el nivel de renta del país, y las tecnologías, que vienen como listas
separadas por punto y coma dentro de una sola celda.
"""),
(CODE, r"""
from src.data_loader_stackoverflow import expandir_multivalor, CATEGORIA_AUSENTE

df = derivar_region(df, ref)
for col in ("YearsCode", "YearsCodePro", "WorkExp"):
    df[f"{col}_num"] = parsear_anios(df[col])
df["salary_log"] = np.log1p(df[TARGET])

# Las categóricas ausentes reciben una categoría propia en vez de imputarse: no
# declarar el sector es un dato, no un hueco que haya que rellenar.
for col in ("EdLevel", "DevType", "OrgSize", "RemoteWork", "Industry", "ICorPM", "Age"):
    df[col] = df[col].fillna(CATEGORIA_AUSENTE)

print("Antes de expandir, una celda de lenguajes se ve así:")
print(" ", df["LanguageHaveWorkedWith"].dropna().iloc[0][:110])

antes_cols = df.shape[1]
for col in ("LanguageHaveWorkedWith", "DatabaseHaveWorkedWith", "PlatformHaveWorkedWith"):
    df, nuevas = expandir_multivalor(df, col)
    print(f"{col:28} → {len(nuevas)} columnas binarias")
print(f"\nLa tabla pasa de {antes_cols} a {df.shape[1]} columnas.")
"""),
(CODE, r"""
# Comprobación: ¿mi versión paso a paso coincide con la del módulo?
from src.data_loader_stackoverflow import cargar_encuesta
oficial, _ = cargar_encuesta(anio="2023")
print(f"A mano:      {len(df):,} × {df.shape[1]}")
print(f"Del módulo:  {len(oficial):,} × {oficial.shape[1]}")
print("Coinciden." if len(df) == len(oficial) and df.shape[1] == oficial.shape[1]
      else "NO coinciden: hay un paso que se me ha escapado.")
"""),
(MD, """
Coincide, 37,622 filas y 133 columnas por los dos caminos. Bien, porque significa
que no me he saltado ningún paso al hacerlo a mano.

De 89,184 me quedo con 37,622, el 42.2 %. Es menos de la mitad y conviene decirlo
claro: esto **no** es una muestra representativa de los desarrolladores del
mundo. Es quien responde la encuesta de Stack Overflow, declara su sueldo,
trabaja por cuenta ajena y vive en uno de los 78 países con suficientes
respuestas.
"""),

# ---------------------------------------------------------------------------
(MD, """
## 5. Mirar la muestra ya limpia
"""),
(CODE, r"""
from scripts.generate_figures import estilo, fig_distribucion, fig_composicion
estilo()

print(f"Mediana  ${df[TARGET].median():>10,.0f}")
print(f"Media    ${df[TARGET].mean():>10,.0f}")
print(f"Asimetría {df[TARGET].skew():>10.2f}   ·   tras log1p {df['salary_log'].skew():>6.2f}")

figura(fig_distribucion(df, FIGURAS))
"""),
(MD, """
La mediana está en 78,175 dólares y la media en 96,319: la media va casi veinte
mil por encima, que es lo que pasa cuando la cola de la derecha pesa.

En el gráfico de la izquierda se ve el amontonamiento a la izquierda y la cola
larga. En el de la derecha, tras el logaritmo, la forma es mucho más manejable,
aunque no es una campana: hay un bulto a la izquierda que es lo que deja la
asimetría en −0.82.

A partir de aquí trabajo sobre el logaritmo.
"""),
(CODE, r"""
comp = (df.groupby("income_group", observed=True)[TARGET]
          .agg(N="size", Mediana="median",
               P25=lambda s: s.quantile(.25), P75=lambda s: s.quantile(.75)))
comp["% muestra"] = (comp["N"] / len(df) * 100).round(1)
print(comp.round(0).to_string())

figura(fig_composicion(df, FIGURAS))
"""),
(MD, """
El 85 % de la muestra son países de renta alta, y la mediana de ese grupo (86,743)
es **cinco veces** la de renta media-baja (16,964).

Dos cosas que me apunto. Una, cualquier media global va a estar dominada por los
países ricos. Y dos, si reparto entrenamiento y prueba al azar, con solo un 5 %
de renta media-baja puede tocarme un reparto desigual; mejor estratificar.
"""),

# ---------------------------------------------------------------------------
(MD, """
## 6. Los predictores, y el problema del país

Separo lo que entra al modelo de lo que solo sirve para auditarlo después.
"""),
(CODE, r"""
from src.feature_pipeline_so import preparar_xy, CAT_ALTA, CAT_BAJA, NUMERICAS

X, y, A = preparar_xy(df)
print(f"X: {X.shape[0]:,} filas × {X.shape[1]} columnas")
print(f"y: logaritmo del salario")
print(f"A: {list(A.columns)}  ← fuera del modelo, para auditar después\n")

# Cuento las categorías realmente declaradas: "No declarado" es una etiqueta que
# he puesto yo y contarla como una más inflaría la cardinalidad.
print("Cardinalidad de las categóricas:")
for col in CAT_ALTA + CAT_BAJA:
    if col in X.columns:
        declaradas = X[col][X[col] != CATEGORIA_AUSENTE]
        print(f"  {col:14} {declaradas.nunique():>4} categorías   "
              f"cobertura {len(declaradas)/len(X)*100:>5.1f} %")
"""),
(MD, """
119 columnas entran al modelo. `Country` tiene 78 categorías y `DevType` 33: son
las dos que van a dar problemas, porque si las convierto en variables binarias
añaden más de cien columnas ellas solas.

`Industry` solo la declara el 61.6 % e `ICorPM` el 72 %. No las imputo: les pongo
una categoría «No declarado» y que el modelo decida si le dice algo. Rellenar con
la moda sería inventarme el dato.

El género no está: la edición 2023 ya no lo publica. Habrá que ir a 2022 para eso.
"""),
(CODE, r"""
from src.feature_pipeline_so import construir_preprocesador, nombres_de_variables
from sklearn.model_selection import train_test_split

X_ent, X_pru, y_ent, y_pru, A_ent, A_pru = train_test_split(
    X, y, A, test_size=0.20, random_state=SEMILLA, stratify=df["income_group"])

# Estratifico por nivel de renta y no al azar: con el 85 % de la muestra en un
# solo grupo, una partición aleatoria puede dejar los grupos pequeños muy
# desigualmente repartidos.
print(f"Entrenamiento {len(X_ent):,}   ·   prueba {len(X_pru):,}\n")
print("Reparto por nivel de renta (%):")
print(pd.DataFrame({
    "entrenamiento": df.loc[X_ent.index, "income_group"].value_counts(normalize=True) * 100,
    "prueba": df.loc[X_pru.index, "income_group"].value_counts(normalize=True) * 100,
}).round(1).to_string())

for cod in ("target", "onehot"):
    pre = construir_preprocesador(df, cod).fit(X_ent, y_ent)
    print(f"\nCodificación «{cod}»: {X.shape[1]} columnas de entrada "
          f"→ {len(nombres_de_variables(pre))} tras codificar")
"""),
(MD, """
El reparto por nivel de renta queda idéntico entre entrenamiento y prueba, que es
justo lo que buscaba con la estratificación.

Y aquí se ve el problema de `Country` y `DevType` en números: las mismas 119
columnas de entrada dan **162** con codificación por objetivo y **261** con
codificación disyuntiva. Cien columnas de diferencia, casi todas categorías de
país. Voy a probar las dos y comparar, porque no tengo claro cuál conviene.
"""),

# ---------------------------------------------------------------------------
(MD, """
## 7. ¿Contra qué comparo?

Antes de entrenar nada conviene saber cuánto vale no hacer nada. Si el modelo no
le gana a responder siempre la media, no sirve.
"""),
(CODE, r"""
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from src.metrics import compute_metrics
from src.model_registry import build_models

modelos = {
    "Baseline_media": DummyRegressor(strategy="mean"),
    "Baseline_mediana": DummyRegressor(strategy="median"),
    "Ridge": Ridge(alpha=1.0),
    **build_models(),
}
print("Voy a evaluar:", ", ".join(modelos))

for nombre in ("Baseline_media", "Baseline_mediana"):
    t = Pipeline([("preprocesador", construir_preprocesador(df, "target")),
                  ("modelo", modelos[nombre])]).fit(X_ent, y_ent)
    m = compute_metrics(y_pru, t.predict(X_pru))
    print(f"\n{nombre}:  R² = {m['R2']:.4f}   MAE = ${m['MAE_USD']:,.0f}")
"""),
(MD, """
R² de −0.0002 y −0.0057. Negativos, como debe ser: responder siempre la media no
explica nada, y responder la mediana es incluso un poco peor porque la
distribución no es simétrica.

El dato que me quedo es el MAE: **49,826 dólares** de error medio sin usar
ninguna variable. Ese es el listón.
"""),

# ---------------------------------------------------------------------------
(MD, """
## 8. Validación cruzada

Una sola partición no me dice si la diferencia entre dos modelos es real o es la
suerte del reparto. Uso validación cruzada repetida, sobre las mismas
particiones para todos, para poder compararlos de dos en dos.
"""),
(CODE, r"""
import time
from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold

N_PARTICIONES, N_REPETICIONES = 5, 4
particionador = RepeatedStratifiedKFold(n_splits=N_PARTICIONES,
                                        n_repeats=N_REPETICIONES,
                                        random_state=SEMILLA)
estratos = df.loc[X_ent.index, "income_group"].astype(str)

filas, t0 = [], time.perf_counter()
for i, (tr, va) in enumerate(particionador.split(X_ent, estratos)):
    for codificacion in ("target", "onehot"):
        for nombre, modelo in modelos.items():
            # El preprocesador se reajusta dentro de cada partición: la
            # codificación por objetivo usa la y, y ajustarla fuera dejaría
            # entrar información del conjunto de validación.
            t = Pipeline([("preprocesador", construir_preprocesador(df, codificacion)),
                          ("modelo", clone(modelo))])
            t.fit(X_ent.iloc[tr], y_ent[tr])
            m = compute_metrics(y_ent[va], t.predict(X_ent.iloc[va]))
            m.update({"modelo": nombre, "codificacion": codificacion, "particion": i})
            filas.append(m)
    if (i + 1) % N_PARTICIONES == 0:
        print(f"  repetición {(i+1)//N_PARTICIONES}/{N_REPETICIONES}  "
              f"({time.perf_counter()-t0:.0f} s)")

cv = pd.DataFrame(filas)
cv.to_csv(SALIDA / "cv_by_encoding.csv", index=False)
print(f"\n{len(cv)} ajustes en {time.perf_counter()-t0:.0f} segundos")
"""),
(CODE, r"""
resumen_cv = (cv.groupby(["modelo", "codificacion"])
                .agg(R2=("R2", "mean"), desv=("R2", "std"), MAE_USD=("MAE_USD", "mean"))
                .round(4).sort_values("R2", ascending=False))
print(resumen_cv.to_string())

from scripts.generate_figures import fig_modelos
figura(fig_modelos(FIGURAS, SALIDA))
"""),
(MD, """
XGBoost con codificación por objetivo queda primero, 0.7863.

Lo que no esperaba es lo pegados que están: entre el primero y el quinto hay tres
milésimas de R², y la desviación típica entre particiones es de seis milésimas.
O sea que **la diferencia entre los cuatro ensamblados es más pequeña que lo que
varían ellos mismos de una partición a otra**. Quedarme con el primero por su
media y decir que es el mejor sería colarme.

Ridge se queda en 0.7544, tres puntos por debajo. Y el MAE baja de 49,826 a
24,890: la mitad del error de no hacer nada.
"""),

# ---------------------------------------------------------------------------
(MD, """
## 9. El conjunto de prueba

Hasta aquí no lo he tocado.
"""),
(CODE, r"""
tuberias, filas = {}, []
for codificacion in ("target", "onehot"):
    for nombre, modelo in modelos.items():
        t = Pipeline([("preprocesador", construir_preprocesador(df, codificacion)),
                      ("modelo", clone(modelo))]).fit(X_ent, y_ent)
        m = compute_metrics(y_pru, t.predict(X_pru))
        m.update({"modelo": nombre, "codificacion": codificacion})
        filas.append(m)
        tuberias[(nombre, codificacion)] = t

prueba = pd.DataFrame(filas)
prueba.to_csv(SALIDA / "test_by_encoding.csv", index=False)
print(prueba.set_index(["modelo", "codificacion"])[["R2", "MAE_USD", "MedAE_USD", "MAPE"]]
      .round(4).sort_values("R2", ascending=False).to_string())
"""),
(MD, """
En prueba el orden cambia: se pone delante CatBoost con 0.7865 y XGBoost baja al
segundo por tres diezmilésimas.

No lo leo como que CatBoost sea mejor, sino como confirmación de lo de antes: las
diferencias están por debajo del ruido, y 7,525 observaciones bastan para
reordenarlos. Si tuviera que elegir uno para usarlo de verdad miraría el tiempo
de ajuste antes que la cuarta cifra decimal, y ahí CatBoost tarda cuatro veces
más que XGBoost.

Lo importante es que prueba y validación cruzada se parecen. Si hubiera
sobreajuste, aquí se habría notado.
"""),
(CODE, r"""
from scripts.generate_figures import fig_residuos
figura(fig_residuos(FIGURAS, df, SALIDA))
"""),
(MD, """
Los residuos están centrados y no se abren ni se cierran con el valor estimado,
que es lo que quería comprobar.

Lo que sí se ve es que no son simétricos: hay más masa por debajo. El modelo se
ajusta sobre el logaritmo y al deshacer la transformación con la exponencial el
resultado queda por debajo de la media real. Es un efecto conocido de trabajar en
logaritmos, no es que el modelo esté torcido, pero conviene tenerlo presente
porque aparecerá otra vez cuando mire el sesgo por grupos.
"""),

# ---------------------------------------------------------------------------
(MD, """
## 10. ¿Las diferencias son de verdad?

Tengo veinte medidas de R² por configuración, sobre las mismas particiones. La
prueba de Wilcoxon pareada compara dos configuraciones partición a partición.
"""),
(CODE, r"""
from scipy.stats import wilcoxon

pivote = cv.pivot_table(index="particion", columns=["modelo", "codificacion"], values="R2")

# Antes de usarla, una comprobación que casi me cuesta un disgusto: con pocas
# particiones la prueba no puede dar un valor p pequeño por mucho que la
# diferencia sea grande, porque no hay combinaciones de signos suficientes.
for n in (5, 10, 20):
    x = np.arange(1, n + 1)
    print(f"  con {n:>2} pares, el menor valor p posible es {wilcoxon(x, -x).pvalue:.2g}")
"""),
(MD, """
Menos mal que lo he mirado antes de usarla.

**Con cinco particiones el menor valor p que la prueba puede dar es 0.062**, por
encima del 0.05. Da igual lo grande que sea la diferencia: con cinco pares no hay
suficientes combinaciones de signos para bajar de ahí, y ninguna comparación
saldría significativa nunca.

Con veinte pares baja a 1.9 × 10⁻⁶. Por eso he repetido la validación cruzada
cuatro veces en vez de hacerla una sola.
"""),
(CODE, r"""
def comparar(a, b, familia):
    x, y_ = pivote[a], pivote[b]
    stat, p = wilcoxon(x, y_)
    return {"familia": familia, "config_a": f"{a[0]}/{a[1]}", "config_b": f"{b[0]}/{b[1]}",
            "n_pares": len(x), "media_a": round(x.mean(), 4), "media_b": round(y_.mean(), 4),
            "diferencia": round((x - y_).mean(), 4), "p_valor": p}

reales = [m for m in modelos if not m.startswith("Baseline")]
comparaciones = []
for m in reales:                                    # ¿importa la codificación?
    comparaciones.append(comparar((m, "target"), (m, "onehot"), "codificacion"))
for c in ("target", "onehot"):                      # ¿le gano a no hacer nada?
    for m in reales:
        for base in ("Baseline_media", "Baseline_mediana"):
            comparaciones.append(comparar((m, c), (base, c), "contra_baseline"))
for c in ("target", "onehot"):                      # ¿hace falta un ensamblado?
    for m in [r for r in reales if r != "Ridge"]:
        comparaciones.append(comparar(("Ridge", c), (m, c), "lineal_vs_ensamblado"))

# Holm-Bonferroni dentro de cada familia: comparo muchas veces y sin corregir
# alguna saldría significativa por puro azar.
def holm(bloque):
    bloque = bloque.sort_values("p_valor").reset_index(drop=True)
    bloque["umbral"] = [0.05 / (len(bloque) - k) for k in range(len(bloque))]
    bloque["significativo"] = bloque["p_valor"] < bloque["umbral"]
    fallidas = bloque.index[~bloque["significativo"]]
    if len(fallidas):
        bloque.loc[fallidas.min():, "significativo"] = False
    return bloque

contrastes = pd.concat([holm(b) for _, b in pd.DataFrame(comparaciones).groupby("familia")],
                       ignore_index=True)
contrastes.to_csv(SALIDA / "wilcoxon_tests.csv", index=False)

print("¿Le gano a las líneas base?")
cb = contrastes[contrastes.familia == "contra_baseline"]
print(f"  {cb.significativo.sum()} de {len(cb)} comparaciones significativas\n")

print("¿Importa la estrategia de codificación?")
print(contrastes[contrastes.familia == "codificacion"]
      [["config_a", "config_b", "media_a", "media_b", "diferencia", "p_valor", "significativo"]]
      .to_string(index=False))
"""),
(MD, """
Las veinte comparaciones contra las líneas base salen significativas. Tampoco es
un resultado que sorprenda: la diferencia entre R² de 0.78 y R² de 0.00 no
necesitaba una prueba estadística.

Lo de la codificación es más interesante y no contesta lo que yo pensaba
preguntar. No es que una sea mejor: **depende del modelo**. La codificación por
objetivo gana en XGBoost y CatBoost, **pierde** en Ridge, y en LightGBM y
HistGradientBoosting la diferencia no es distinguible del ruido.

Que pierda en Ridge tiene sentido cuando lo piensas: un modelo lineal no puede
hacer gran cosa con una variable que resume el país en un solo número, mientras
que un árbol puede partirla por donde quiera.

Y conviene mirar la columna de diferencias antes de emocionarse: la mayor es de
0.0040 de R². Es significativa porque se repite en las veinte particiones, no
porque sea grande.
"""),
(CODE, r"""
print("¿Hace falta un ensamblado, o basta el modelo lineal?")
print(contrastes[contrastes.familia == "lineal_vs_ensamblado"]
      [["config_a", "config_b", "diferencia", "p_valor", "significativo"]]
      .to_string(index=False))
"""),
(MD, """
Aquí no hay ambigüedad: los ensamblados le ganan a Ridge en las ocho
comparaciones, con diferencias de entre 0.027 y 0.036 de R². Son un orden de
magnitud mayores que las de la codificación.

O sea que la complejidad del ensamblado sí se paga con exactitud, y la elección
del modelo importa bastante más que cómo codifique las categóricas.
"""),

# ---------------------------------------------------------------------------
(MD, """
## 11. ¿De qué depende lo que estima?

Me quedo con la configuración que mejor salió en validación cruzada y calculo
los valores SHAP: cuánto empuja cada variable, en cada persona concreta, la
estimación arriba o abajo.
"""),
(CODE, r"""
from src.explainability import explicar

mejor = resumen_cv.index[0]
print("Elegida:", mejor[0], "con codificación", mejor[1])

tuberia = tuberias[mejor]
shap_res = explicar(tuberia, X_ent, X_pru,
                    nombres=nombres_de_variables(tuberia.named_steps["preprocesador"]),
                    directorio=str(SALIDA), etiqueta=f"{mejor[0]}_{mejor[1]}_2023")
print(f"\nExplicador: {shap_res['estimador']} sobre {shap_res['n_explicadas']:,} observaciones")
"""),
(CODE, r"""
from scripts.generate_figures import fig_shap
figura(fig_shap(FIGURAS, SALIDA, etiqueta=f"{mejor[0]}_{mejor[1]}_2023"))
"""),
(MD, """
El país domina, y no por poco.

Me esperaba que la experiencia pesara más. La segunda variable son los años de
experiencia profesional, pero a menos de la mitad de la contribución del país.
"""),
(CODE, r"""
imp = pd.read_csv(SALIDA / f"shap_summary_{mejor[0]}_{mejor[1]}_2023.csv")

# Comparo el orden que da SHAP con el de la importancia por impureza, que es la
# que los modelos de árboles traen de serie y la que suele publicarse.
from scipy.stats import spearmanr
comunes = imp.dropna(subset=["importancia_impureza"])
rho, p = spearmanr(comunes["shap_medio_abs"], comunes["importancia_impureza"])
print(f"Correlación de Spearman entre ambas ordenaciones: {rho:.3f}\n")

comparacion = comunes.head(10)[["variable", "shap_medio_abs", "importancia_impureza"]].copy()
comparacion["puesto SHAP"] = range(1, len(comparacion) + 1)
comparacion["puesto impureza"] = (comunes["importancia_impureza"]
                                  .rank(ascending=False).head(10).astype(int).values)
print(comparacion.round(4).to_string(index=False))
"""),
(MD, """
Esto merece pararse.

La correlación entre las dos ordenaciones es 0.749: alta, pero muy lejos de 1. Y
mirando la tabla se ve dónde se rompen.

`income_group_High income` es **primera** por impureza y **tercera** por SHAP.
`YearsCode_num` es sexta por SHAP y **trigésimo primera** por impureza.
`DevType`, cuarta por SHAP, es la diecisiete por impureza.

Son dos cosas distintas. La importancia por impureza mide cuánto usa el modelo
una variable para partir; SHAP mide cuánto mueve la estimación. Una variable
puede aparecer en muchos cortes y mover poco, o al revés.

Lo anoto porque es fácil publicar la de impureza llamándola SHAP: la trae el
modelo de serie, sale en una línea, y a simple vista parece lo mismo. No lo es.
"""),
(CODE, r"""
# Agrupo las variables por bloques para ver el peso de cada tipo de información.
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

por_bloque = (imp.assign(bloque=imp["variable"].map(bloque))
                 .groupby("bloque")["shap_medio_abs"].agg(["sum", "size"])
                 .sort_values("sum", ascending=False))
por_bloque["% del total"] = (por_bloque["sum"] / por_bloque["sum"].sum() * 100).round(1)
print(por_bloque.round(4).to_string())
"""),
(MD, """
Cuatro variables geográficas se llevan el 35 % de la contribución. Las 106
columnas de tecnologías, juntas, el 26 %.

Y el bloque demográfico se queda en el **1.5 %** con siete variables: la edad y lo
que la acompaña casi no intervienen. Conviene recordarlo cuando mire la equidad.

Dicho de otro modo: dónde vives pesa más que lo que sabes hacer. No es una
conclusión cómoda pero es lo que dicen los números.
"""),

# ---------------------------------------------------------------------------
(MD, """
## 12. ¿El efecto de una variable depende de otra?

La importancia por variable da un solo número por variable, así que no puede
contestar a esto. Los valores SHAP sí, porque dan una atribución por persona.
"""),
(CODE, r"""
from run_interacciones import fuerza_interaccion

datos = np.load(SALIDA / f"shap_values_{mejor[0]}_{mejor[1]}_2023.npz", allow_pickle=True)
valores, nombres_var = datos["shap_values"], list(datos["feature_names"])

# El explicador guarda qué filas usó; hay que quedarse con esas mismas y en el
# mismo orden, o los valores SHAP quedarían emparejados con otras personas.
X_t = np.asarray(tuberia.named_steps["preprocesador"].transform(X_pru), dtype=float)
X_t = X_t[datos["indices"]]

top = imp.head(12)["variable"].tolist()
pares = []
for vi in top:
    for vj in top:
        if vi == vj:
            continue
        pares.append({"contribución de": vi, "varía según": vj,
                      "fuerza": round(fuerza_interaccion(
                          valores, X_t, nombres_var.index(vi), nombres_var.index(vj)), 4)})

inter = pd.DataFrame(pares).sort_values("fuerza", ascending=False)
inter.to_csv(SALIDA / "interacciones.csv", index=False)
print(inter.head(10).to_string(index=False))
"""),
(MD, """
Los tres primeros pares no me sirven y conviene verlos antes de seguir.

`income_group_High income` con `Country` es lo mismo dos veces: el nivel de renta
se deriva del país. Y las dos medidas de antigüedad entre sí, igual. Cuando dos
variables miden lo mismo, el modelo reparte la atribución entre ellas de forma
inestable, y esta medida lee ese vaivén como si fuera una interacción.
"""),
(CODE, r"""
# Los pares de arriba incluyen variables que miden lo mismo. Las tres medidas de
# antigüedad están correlacionadas entre sí, y el nivel de renta se deriva del
# país, así que su "interacción" es solo que el modelo reparte la atribución
# entre ellas de forma inestable. Los quito.
redundantes = [
    {"YearsCode_num", "YearsCodePro_num", "WorkExp_num"},
    {"Country", "income_group_High income"},
]
def es_redundante(f):
    return any({f["contribución de"], f["varía según"]} <= grupo for grupo in redundantes)

print(f"Correlación entre las medidas de antigüedad: "
      f"{X[['YearsCode_num', 'YearsCodePro_num']].corr().iloc[0, 1]:.3f}\n")

sustantivas = inter[~inter.apply(es_redundante, axis=1)]
print(sustantivas.head(8).to_string(index=False))
"""),
(MD, """
Quitando los pares redundantes, la interacción más fuerte es la del nivel de
renta del país con los años de experiencia, por los dos lados.

Traducido: **cuánto te paga la experiencia depende del mercado en el que estés**.
No es una tecnicidad, es una afirmación con contenido económico, y quiero verla.
"""),
(CODE, r"""
from scripts.generate_figures import fig_interaccion_experiencia
figura(fig_interaccion_experiencia(FIGURAS, SALIDA, df))
"""),
(MD, """
Ahí está, y es más clara de lo que esperaba.

Los primeros años son iguales en los tres grupos: la contribución es negativa
(empezar cobra menos que la media, evidentemente) y prácticamente la misma en
todas partes. A partir del décimo año las curvas se separan. En los países de
renta media-alta la experiencia sigue sumando; en los de renta alta la curva se
aplana antes. Hacia los veinte años de experiencia, 0.245 frente a 0.172.

Lo que me interesa de esto es que **ninguna medida de importancia por variable
puede decirlo**. Le daría a la experiencia un solo número, promediando tres
trayectorias distintas y perdiendo justo lo que las diferencia.
"""),

# ---------------------------------------------------------------------------
(MD, """
## 13. ¿Funciona igual para todos?

El R² de 0.79 es una media sobre todo el conjunto de prueba. Una media puede
esconder que el modelo funcione muy bien para la mayoría y mal para una minoría,
así que hay que mirarlo por grupos.
"""),
(CODE, r"""
pred = tuberia.predict(X_pru)

# Todo en dólares, no en logaritmos: un error de 0.3 en escala logarítmica no
# significa nada, y además la comparación entre grupos sería engañosa.
real_usd, pred_usd = np.expm1(y_pru), np.expm1(pred)

por_renta = pd.DataFrame({"grupo": A_pru["income_group"].values,
                          "real": real_usd, "pred": pred_usd})
tabla = por_renta.groupby("grupo").apply(lambda g: pd.Series({
    "N": len(g),
    "Mediana real": g["real"].median(),
    "MAE": (g["pred"] - g["real"]).abs().mean(),
}), include_groups=False)
print(tabla.round(0).to_string())
print(f"\nRazón entre el mayor y el menor MAE: "
      f"{tabla['MAE'].max() / tabla['MAE'].min():.2f}")
"""),
(MD, """
El 0.79 de R² era una media, y aquí se ve lo que escondía.

El error en los países de renta alta es de 26,286 dólares y en los de renta
media-baja de 9,846: casi tres veces más en los ricos. Leído tal cual, el modelo
funcionaría mucho peor con los países ricos.

No me lo creo. Los salarios de renta alta son cinco veces mayores, así que es
normal que el error en dólares también lo sea. Comparar errores absolutos entre
grupos que cobran cosas tan distintas no puede estar bien.
"""),
(CODE, r"""
# Divido el error de cada grupo por su propia mediana.
tabla["MAE / mediana"] = tabla["MAE"] / tabla["Mediana real"]
print(tabla.round(3).to_string())
print(f"\nRazón con el error absoluto: {tabla['MAE'].max() / tabla['MAE'].min():.2f}"
      f"   peor: {tabla['MAE'].idxmax()}")
print(f"Razón con el error relativo: "
      f"{tabla['MAE / mediana'].max() / tabla['MAE / mediana'].min():.2f}"
      f"   peor: {tabla['MAE / mediana'].idxmax()}")
"""),
(MD, """
Efectivamente, **se da la vuelta**.

En proporción al salario típico de cada sitio, el modelo se equivoca un 29.8 % en
los países de renta alta y un **58.0 %** en los de renta media-baja. El doble. El
grupo peor servido no es el que decía el error absoluto: es justo el contrario.

La razón de disparidad baja de 2.67 a 1.95, pero las dos superan cualquier umbral
razonable, así que la disparidad existe de las dos maneras. Lo que cambia, y es
lo importante, es **a quién perjudica**.

Si hubiera hecho la auditoría con el error en dólares habría escrito la
conclusión contraria a la correcta, y encima habría sonado convincente.
"""),
(CODE, r"""
from src.fairness import auditar, resumir
import json

for atributo in ("income_group", "wb_region", "Age", "EdLevel", "OrgSize"):
    aud = auditar(y_pru, pred, A_pru[atributo])
    (SALIDA / f"fairness_{atributo}_2023.json").write_text(
        json.dumps(aud, ensure_ascii=False, indent=1), encoding="utf-8")
print(resumir(auditar(y_pru, pred, A_pru["wb_region"]), "Región"))
"""),
(CODE, r"""
from scripts.generate_figures import fig_equidad
figura(fig_equidad(FIGURAS, SALIDA, anio="2023"))
"""),
(MD, """
Por región se ve lo mismo y más marcado. América Latina, 58.7 % de error
relativo; Asia del Sur, 54.9 %; Europa y Asia Oriental, entre 25 y 27 %.

Dos cosas más del resumen. La prueba de Kruskal-Wallis da p = 5.9 × 10⁻⁶: la
diferencia entre regiones no es casualidad del reparto. Y la distancia de
Kolmogórov-Smirnov entre Norteamérica y Asia del Sur es 0.995, o sea que las
compensaciones que el modelo predice para una y otra región no se solapan
prácticamente nada.

El sesgo es negativo en todos los grupos sin excepción, que es el efecto de la
exponencial que ya salía en los residuos. Como afecta a todos por igual, no es
una diferencia de trato entre grupos.
"""),

# ---------------------------------------------------------------------------
(MD, """
## 14. El género, que hay que ir a buscar a 2022

La edición 2023 no pregunta por género. La de 2022 sí, así que repito sobre ella
lo imprescindible.
"""),
(CODE, r"""
df22, _ = cargar_encuesta(anio="2022")
print(f"Edición 2022: {len(df22):,} observaciones efectivas")
print(df22["Gender"].value_counts().head(4).to_string())

X22, y22, A22 = preparar_xy(df22)
Xe, Xp, ye, yp, _, Ap = train_test_split(
    X22, y22, A22, test_size=0.20, random_state=SEMILLA, stratify=df22["income_group"])
t22 = Pipeline([("preprocesador", construir_preprocesador(df22, mejor[1])),
                ("modelo", clone(modelos[mejor[0]]))]).fit(Xe, ye)
p22 = t22.predict(Xp)
print(f"\nR² sobre prueba: {compute_metrics(yp, p22)['R2']:.4f}")

aud_g = auditar(yp, p22, Ap["Gender"])
(SALIDA / "fairness_Gender_2022.json").write_text(
    json.dumps(aud_g, ensure_ascii=False, indent=1), encoding="utf-8")
print()
print(resumir(aud_g, "Género"))
"""),
(MD, """
La edición 2022 tiene 28,951 observaciones válidas, bastantes menos que 2023, y
el R² sale 0.7799, muy parecido. Al menos el modelo se comporta igual en las dos.

La tabla sale con once grupos porque la pregunta permite marcar varias opciones a
la vez, y la mayoría de las combinaciones tienen dos o tres personas. La función
las marca como descriptivas y las deja fuera de la inferencia. Aun así ensucian
la lectura, así que me quedo con los cuatro que pasan de treinta.
"""),
(CODE, r"""
# Junto el error del modelo con el salario real, que son las dos cosas que hay
# que mirar a la vez.
g = pd.DataFrame(aud_g["por_grupo"]).T
g = g[g["n"].astype(int) >= 30]
mediana = (pd.DataFrame({"g": Ap["Gender"].values, "y": np.expm1(yp)})
             .groupby("g")["y"].median())
print(pd.DataFrame({
    "N": g["n"].astype(int),
    "Mediana real (USD)": mediana.reindex(g.index).round(0),
    "MAE relativo": g["mae_relativo"].astype(float).round(3),
}).sort_values("N", ascending=False).to_string())
"""),
(MD, """
Esto es lo que quería ver, y contesta a la pregunta que motivaba toda la
auditoría.

Hombres, 0.360 de error relativo. Mujeres, 0.362. **Dos milésimas**, sobre 5,310
y 286 observaciones.

Y a la vez la mediana real de las mujeres está por debajo de la de los hombres.
O sea que la brecha salarial **está en los datos**, pero **no está en el error del
modelo**: el sistema estima igual de bien a unas y a otros. Lo que reproduce es un
mercado desigual, no una desigualdad propia.

Son dos cosas que se confunden con facilidad y llevan a conclusiones opuestas.

Sobre la razón de disparidad de 1.347 que sale en el resumen: no viene de este
contraste, viene de las setenta personas que no declararon género (0.486) y las
cuarenta y siete no binarias (0.413). Si cito ese número suelto estoy diciendo
algo que los datos no dicen.
"""),
(CODE, r"""
# Y cruzando género con región, que es donde la literatura dice que aparece lo
# que el agregado esconde.
lat = A22.loc[Xp.index, "wb_region"] == "Latin America & Caribbean"
gen = Ap["Gender"]
filas = []
for region, mascara in [("América Latina", lat), ("Resto del mundo", ~lat)]:
    for etiqueta, valor in [("Hombres", "Man"), ("Mujeres", "Woman")]:
        sel = mascara & (gen == valor)
        if sel.sum() == 0:
            continue
        real, prd = np.expm1(yp[sel.values]), np.expm1(p22[sel.values])
        filas.append({"Región": region, "Género": etiqueta, "N": int(sel.sum()),
                      "Mediana real": round(float(np.median(real))),
                      "MAE": round(float(np.abs(prd - real).mean())),
                      "MAE relativo": round(float(np.abs(prd - real).mean() / np.median(real)), 3)})
print(pd.DataFrame(filas).to_string(index=False))
"""),
(MD, """
Y cruzando las dos cosas aparece lo que el agregado escondía.

Fuera de América Latina, hombres 0.348 y mujeres 0.361: una diferencia de poco más
de un punto. Dentro de América Latina, hombres 0.547 y mujeres **0.719**:
diecisiete puntos.

Ahora bien, **son diecisiete mujeres**. Con ese número no puedo afirmar nada:
cualquier diferencia de esa magnitud cabe dentro de la variabilidad de una muestra
tan pequeña. Lo dejo como descriptivo y como indicación de por dónde habría que
mirar con más datos.

Quitarlo tampoco me parece bien. Sería esconder justo el cruce que la literatura
señala como el más fácil de perder de vista.
"""),

# ---------------------------------------------------------------------------
(MD, """
## 15. ¿Se puede arreglar la disparidad?

Lo primero que se suele proponer es reequilibrar la muestra. Pruebo tres formas.
"""),
(CODE, r"""
from run_mitigacion import pesos_inversos, remuestrear

rng = np.random.default_rng(SEMILLA)
g_ent = df.loc[X_ent.index, "income_group"].astype(str)
resultados_mit = []

for estrategia in ("sin_mitigacion", "reponderacion", "submuestreo", "sobremuestreo"):
    t = Pipeline([("preprocesador", construir_preprocesador(df, mejor[1])),
                  ("modelo", clone(modelos[mejor[0]]))])
    if estrategia == "reponderacion":
        t.fit(X_ent, y_ent, modelo__sample_weight=pesos_inversos(g_ent))
        Xm, ym = X_ent, y_ent
    elif estrategia == "sin_mitigacion":
        t.fit(X_ent, y_ent); Xm, ym = X_ent, y_ent
    else:
        # remuestrear devuelve índices, no la tabla ya remuestreada.
        idx = remuestrear(X_ent, y_ent, g_ent, estrategia, rng)
        Xm, ym = X_ent.iloc[idx], y_ent[idx]
        t.fit(Xm, ym)
    p = t.predict(X_pru)
    a = auditar(y_pru, p, A_pru["income_group"])
    m = compute_metrics(y_pru, p)
    resultados_mit.append({
        "estrategia": estrategia, "n_entrenamiento": len(Xm),
        "R2": round(m["R2"], 4), "MAE_log": round(m["MAE_log"], 4),
        "MAE_USD": round(m["MAE_USD"]), "MAPE": round(m["MAPE"], 2),
        "razon_disparidad_renta": round(a["agregados"]["razon_disparidad_relativa"], 3),
        "razon_disparidad_region": round(
            auditar(y_pru, p, A_pru["wb_region"])["agregados"]["razon_disparidad_relativa"], 3),
        "cv_relativo_renta": round(a["agregados"]["coeficiente_variacion_relativo"], 3),
        "peor_grupo": a["agregados"]["grupo_peor_servido_relativo"],
        "relativo_por_grupo": {k: round(v["mae_relativo"], 4)
                               for k, v in a["por_grupo"].items()},
    })
    print(f"{estrategia:16} N={len(Xm):>6,}  R²={m['R2']:.4f}  "
          f"razón={a['agregados']['razon_disparidad_relativa']:.3f}")

mit = pd.DataFrame(resultados_mit)
mit.drop(columns="relativo_por_grupo").to_csv(SALIDA / "mitigacion_resultados.csv", index=False)
"""),
(MD, """
Ninguna de las tres funciona.

La reponderación y el sobremuestreo mueven la razón de disparidad de 1.947 a 1.932
y 1.933: una centésima y media. El submuestreo sí la baja de verdad, a 1.815, pero
pagando 0.036 de R² y **tirando el 84 % de los datos de entrenamiento** (de 30,097
a 4,842).

O sea que lo único que mueve la aguja lo hace empeorando el modelo. Mal negocio.
"""),
(CODE, r"""
# Lo que de verdad importa no es la razón, sino si los grupos pequeños mejoran.
detalle = pd.DataFrame({r["estrategia"]: r["relativo_por_grupo"] for r in resultados_mit})
print((detalle * 100).round(1).to_string())
"""),
(MD, """
Y aquí está el detalle que lo explica, que es lo que de verdad importa.

**El error de los grupos pequeños no mejora con ninguna estrategia. Empeora
ligeramente con todas.** Renta media-baja pasa de 58.0 % a 58.6, 58.8 y 59.2 %.

Entonces, ¿por qué baja la razón de disparidad con el submuestreo? Porque el
grupo mayoritario empeora más (de 29.8 a 32.6 %). La razón mejora **estropeando al
grande**, no arreglando a los pequeños. Es una mejora de la métrica que no
corresponde a ninguna mejora real para nadie.
"""),
(CODE, r"""
# Si no es un problema de representación, ¿de qué es? Miro cuánto varía el
# salario DENTRO de cada país, que es lo que el modelo tendría que explicar.
disp = (df.groupby("income_group", observed=True)
          .apply(lambda g: g.groupby("Country")["salary_log"].std().median(),
                 include_groups=False)
          .rename("desv. típica intra-país").to_frame())
disp["razón frente a renta alta"] = (disp["desv. típica intra-país"] /
                                     disp.loc["High income", "desv. típica intra-país"])
print(disp.round(3).to_string())

from scripts.generate_figures import fig_mitigacion
figura(fig_mitigacion(FIGURAS, SALIDA))
"""),
(MD, """
Aquí está el porqué, y me parece el hallazgo más útil de todo el trabajo.

La desviación típica del salario **dentro de cada país** es 0.477 en los de renta
alta y 0.781 en los de renta media-baja: un 64 % mayor. Dos personas del mismo
país, con perfiles parecidos según lo que la encuesta pregunta, cobran cosas mucho
más distintas en un país de renta media-baja.

Es decir: el mayor error relativo del modelo en esas regiones **no es que las trate
peor, es que el salario allí es menos predecible con las variables que hay**.
Ningún remuestreo puede arreglar aquello para lo que no hay señal en los datos.

Y la comprobación es barata: comparar la varianza de la variable objetivo dentro de
cada grupo antes de ponerse a remuestrear. Si hubiera empezado por ahí me habría
ahorrado las tres pruebas.
"""),

# ---------------------------------------------------------------------------
(MD, """
## 16. La inteligencia artificial, en la edición 2025

La edición 2025 es la primera que pregunta por el uso de herramientas de IA.
"""),
(CODE, r"""
df25, _ = cargar_encuesta(anio="2025")
print(f"Edición 2025: {len(df25):,} observaciones efectivas")
print(df25["AISelect"].value_counts().to_string())
"""),
(CODE, r"""
# Primero la asociación en bruto: ¿cobra distinto quien usa IA?
from run_analysis import analizar_ia
ia = analizar_ia(df25, SALIDA)
print(ia[ia.variable == "AISelect"][["categoria", "n", "mediana_usd"]].to_string(index=False))
"""),
(MD, """
Sale lo contrario de lo que uno esperaría: **quien más usa IA es quien menos
cobra**. Mediana de 81,685 dólares con uso diario, frente a 87,550 entre quienes
no la usan ni piensan usarla.

No me lo tomo en serio todavía. Quien usa IA a diario puede ser sistemáticamente
más joven, o estar en otro país, o en otro tipo de empresa. La asociación bruta no
descarta nada de eso.
"""),
(CODE, r"""
# Ahora la pregunta de verdad: ¿aporta algo la IA una vez que el modelo ya sabe
# el país, la experiencia y el rol?
X25, y25, A25 = preparar_xy(df25)
Xe25, Xp25, ye25, yp25 = train_test_split(
    X25, y25, test_size=0.20, random_state=SEMILLA, stratify=df25["income_group"])
t25 = Pipeline([("preprocesador", construir_preprocesador(df25, mejor[1])),
                ("modelo", clone(modelos[mejor[0]]))]).fit(Xe25, ye25)
res25 = explicar(t25, Xe25, Xp25,
                 nombres=nombres_de_variables(t25.named_steps["preprocesador"]),
                 directorio=str(SALIDA), etiqueta=f"{mejor[0]}_{mejor[1]}_2025")

imp25 = pd.read_csv(SALIDA / f"shap_summary_{mejor[0]}_{mejor[1]}_2025.csv")
ia_vars = imp25[imp25["variable"].str.startswith(("AISelect", "AIThreat", "AIAgents", "LearnCodeAI"))]
print(f"R² sobre prueba (2025): {compute_metrics(yp25, t25.predict(Xp25))['R2']:.4f}\n")
print(f"Variables de IA tras codificar: {len(ia_vars)}")
print(f"Contribución conjunta: {ia_vars['shap_medio_abs'].sum():.4f}")
print(f"Contribución del país: {imp25[imp25.variable.str.startswith('Country')]['shap_medio_abs'].sum():.4f}")
print(f"\nLa de mayor peso entre las de IA ocupa el puesto "
      f"{int(imp25['variable'].tolist().index(ia_vars.iloc[0]['variable'])) + 1} de {len(imp25)}.")
"""),
(CODE, r"""
from scripts.generate_figures import fig_ia
figura(fig_ia(FIGURAS, SALIDA))
"""),
(MD, """
Condicionando por lo demás, la asociación se deshace.

Las 23 variables de IA juntas suman 0.0653 de contribución. El país solo, 0.4312:
**seis veces y media más que las de IA todas juntas**. La de más peso entre ellas
es la decimosexta de 192.

Así que la relación del primer gráfico era composición de la muestra, no efecto de
la IA. Quien la usa a diario tiene un perfil distinto, y es ese perfil el que
explica el sueldo.

Dicho esto, y para no pasarme: esto son datos de un corte temporal, no un
experimento. Que la adopción de IA no aporte al modelo no significa que usarla no
sirva para nada, solo que no distingue salarios una vez que ya sabes el país, la
experiencia y el rol.
"""),

# ---------------------------------------------------------------------------
(MD, """
## 17. Recapitulando
"""),
(CODE, r"""
print(f"Muestra 2023          {len(df):>10,} observaciones, {df['Country'].nunique()} países")
print(f"Mejor configuración   {mejor[0]} / {mejor[1]}")
print(f"R² validación cruzada {resumen_cv['R2'].max():>10.4f}")
print(f"R² conjunto de prueba {prueba['R2'].max():>10.4f}")
print(f"MAE                   {prueba.loc[prueba.R2.idxmax(), 'MAE_USD']:>10,.0f} USD")
print(f"Peso del país         {por_bloque.loc['Geográfico', '% del total']:>10.1f} % de la contribución total")
print(f"Peso del capital humano {por_bloque.loc['Capital humano', '% del total']:>8.1f} %")
print(f"Error relativo, renta alta       {tabla.loc['High income', 'MAE / mediana']:>6.1%}")
print(f"Error relativo, renta media-baja {tabla.loc['Lower middle income', 'MAE / mediana']:>6.1%}")
"""),
(MD, """
Resumiendo lo que ha salido.

El modelo explica el 79 % de la variación del logaritmo del salario y se equivoca
de media en unos 24,760 dólares, la mitad que responder siempre la media. La
diferencia con las líneas base es significativa en las veinte particiones.

Lo que manda es el país: el 35 % de la contribución con cuatro variables, frente al
22 % del capital humano con quince. Dónde vives pesa más que tu trayectoria.

La codificación de las categóricas de alta cardinalidad no tiene un ganador: gana
la codificación por objetivo en unos modelos, pierde en otros.

Y sobre equidad, tres cosas. El modelo **no** estima peor a las mujeres, pese a que
la brecha salarial está en los datos. **Sí** estima bastante peor, en términos
relativos, a los países de menor renta: 58 % frente a 30 %. Y eso **no se arregla**
reequilibrando la muestra, porque no viene de la representación sino de que el
salario es allí intrínsecamente más disperso.

Lo que este cuaderno no contesta: si las decisiones que he ido tomando por el
camino (los seis filtros, el umbral de plausibilidad, medir el error en relativo)
son las correctas. Son decisiones discutibles y las he dejado a la vista
precisamente para que se puedan discutir.
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
        "nbformat": 4, "nbformat_minor": 5,
    }
    DESTINO.parent.mkdir(exist_ok=True)
    DESTINO.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{DESTINO.relative_to(RAIZ)}: {len(CELDAS)} celdas "
          f"({sum(1 for t, _ in CELDAS if t == CODE)} de código)")


if __name__ == "__main__":
    main()
