# Auditoría del proyecto `tech-salary-prediction` y hoja de ruta hacia tesis

Fecha de análisis: 2026-09-03
Alcance: se revisaron el historial de git, los 4 notebooks (`salary_prediction.ipynb` → `v2` → `v3` → `v4`), los CSV de resultados, `feature_importance.csv`, `paper_hallazgos.md`, y el texto completo de ambos Word (versión inicial extendida y versión reducida publicada). El Excel `tech_jobs_salaries.xlsx` se inspeccionó solo a nivel de metadatos (shared strings, cardinalidad de columnas), sin cargar las 200,000 filas.

---

## 1. Qué hay en el repo

| Archivo | Rol |
|---|---|
| `salary_prediction.ipynb` | v1: pipeline con RNN/LSTM/GRU/CNN-LSTM sobre features one-hot. Descartado. |
| `salary_prediction_v2.ipynb` | v2: iteración intermedia sobre features. |
| `salary_prediction_v3.ipynb` | v3: pipeline "clásico" — `MeanTargetEncoder` propio para `country/job_title/primary_skill/secondary_skill`, sin embeddings de texto, sin XGBoost/LightGBM/CatBoost/Stacking. Exporta `results_cv.csv`, `results_test.csv`, `feature_importance.csv` (cell 31). |
| `salary_prediction_v4.ipynb` / `.html` | v4: pipeline "avanzado" — SBERT para texto, `OneHotEncoder` para `country` (ya no target encoding), XGBoost/LightGBM/CatBoost, Optuna, `StackingRegressor`. **No exporta ningún CSV** y sus últimas 4 celdas (40-43) están vacías. |
| `paper_hallazgos.md` | Resumen de resultados "v4", con tablas que en realidad no coinciden con lo que v4 produce (ver §2). |
| `tech_salary_predictios_versioninicial_ES.docx` | Paper extendido (borrador). |
| `tech_salary_predictions_reducido.docx` | Paper recortado — es el que se envió a publicar. |
| `tech_jobs_salaries.xlsx` | Dataset, 200,000 filas. |

---

## 2. Hallazgos críticos (código vs. lo que dice el paper)

Esto es lo más importante del análisis: **el paper describe un pipeline que no es exactamente el que produjo los números que reporta**, y contiene afirmaciones que no están respaldadas por ningún artefacto del repo.

### 2.1 Los CSV citados vienen de v3, pero el método descrito es el de v4
- `results_cv.csv` contiene `LinearRegression, Ridge, Lasso, ElasticNet, RandomForest, ExtraTrees, GradientBoosting, HistGradientBoosting, MLP, Baseline_Mean, Baseline_Median` — exactamente el diccionario `models` de **v3** (notebook v3, cell ~17). v4 nunca calcula estos 11 modelos ni tiene `Lasso/ElasticNet/RandomForest/ExtraTrees/MLP` en su código.
- `feature_importance.csv` (top feature: `country_target` con 0.667) usa el sufijo `_target`, propio del `MeanTargetEncoder` de **v3** (cell 23: `feature_names.extend([f"{c}_target" for c in cat_high_cardinality...])`). v4 ya no usa target encoding para `country` (usa `OneHotEncoder`), así que v4 no puede producir una columna llamada `country_target`.
- Conclusión: los números y gráficos que ambos Word citan como resultados del sistema "con SBERT + XGBoost/LightGBM/CatBoost + Stacking" en realidad son heredados del pipeline anterior y más simple (v3), que no tiene SBERT ni esos modelos de boosting. **Los dos pipelines nunca se ejecutaron juntos de punta a punta produciendo un único set de resultados coherente.**

### 2.2 Los números de la Tabla 5 (Stacking/Ridge/HGB en test) no son reproducibles
- `results_test.csv` (el único artefacto real de test en el repo) solo tiene `Ridge, HistGradientBoosting, GradientBoosting, RandomForest, MLP, ExtraTrees` — **no tiene Stacking, XGBoost, LightGBM ni CatBoost**, pese a que ambos papers presentan una tabla con "Stacking Ensemble" (RMSE=29,721 USD) como el mejor modelo.
- El valor de Ridge en `results_test.csv` es RMSE=29,720.52 USD — casi idéntico al que el paper atribuye a "Stacking Ensemble" (29,721 USD), pero el paper reporta Ridge con un valor distinto (29,748 USD). Esto sugiere una confusión/copy-paste entre modelos al redactar la tabla, no un resultado real de un `StackingRegressor` entrenado.
- El git log confirma manipulación posterior de `results_test.csv` sin relación clara con una re-ejecución completa: commits `df7d0ff` y `8ecf16a` ("Add duplicate ExtraTrees entry" / "Remove duplicate ExtraTrees entry") muestran edición manual del CSV, no regeneración desde el notebook.
- **Implicación:** hoy no puedes reproducir las cifras publicadas ejecutando el código del repo. Para una tesis esto es inaceptable — un comité pedirá correr el notebook y obtener las mismas cifras.

### 2.3 "SHAP" se menciona en el paper pero nunca se calcula en el código
- Ambos Word dedican una subsección entera a "Análisis de Importancia de Características... valores SHAP promedio absolutos" con una tabla de 20 (o 10) features y afirmaciones específicas en USD ("Estados Unidos... aumento salarial predicho de 15,000-40,000 USD", "cada año adicional de experiencia contribuye aproximadamente 2,500-4,000 USD").
- En ningún notebook (v1 a v4) hay una sola línea que invoque `shap.Explainer`, `TreeExplainer` o similar. La librería `shap` se importa de forma opcional en v4 (cell 2, `try: import shap except: shap=None`) pero **nunca se vuelve a usar**.
- Lo que sí existe es `.feature_importances_` de un modelo de árbol (probablemente `GradientBoosting`) en v3 (cell 23), guardado como `feature_importance.csv`. Eso es importancia de Gini/split-based, **no** valores SHAP. Son conceptos relacionados pero numéricamente distintos, y llamarlos "SHAP" en un paper es un error metodológico que un revisor de tesis detectará.
- Las cifras en USD por país y por año de experiencia ("15,000-40,000 USD", "2,500-4,000 USD") no aparecen en ningún CSV ni celda de salida — no hay evidencia de que se hayan calculado; probablemente son estimaciones narrativas redactadas a mano.
- v4 sí importa `PartialDependenceDisplay` y anuncia en su propia celda markdown inicial "Interpretabilidad: SHAP, PDP, LIME", pero ningún de los tres se ejecuta en el notebook tal como está commiteado.

### 2.4 El `StackingRegressor` real no coincide con la fórmula del paper
- Paper: `ŷ_stacking = β0 + β1·p_Ridge + β2·p_HGB + β3·p_Cat` (combinación lineal de solo 3 predicciones).
- Código real (v4, cell 32): `StackingRegressor(estimators=[...], final_estimator=Ridge(alpha=0.5), passthrough=True)`. `passthrough=True` significa que el meta-modelo también recibe **todas las features originales**, no solo las 3 predicciones. La fórmula del paper es una simplificación incorrecta de lo que el código realmente hace.

### 2.5 Tasas de cambio hardcodeadas sin fecha, pese a que el paper reconoce lo contrario
- v4 cell 9 define `currency_to_usd = {'USD':1.0,'INR':0.012,'EUR':1.08,...}` como constantes fijas en el código, sin fecha de referencia ni fuente.
- La sección "Limitaciones" del paper dice que se usó "normalización monetaria mediante tasas de cambio históricas" y en `paper_hallazgos.md` se sugiere (como pendiente) "documentar parámetros de conversión monetaria con fecha de referencia" — es decir, el propio autor ya identificó que esto falta, pero quedó sin resolver en la versión publicada.

### 2.6 Origen del dataset: la afirmación de fuente es muy probablemente incorrecta
Este es, en mi opinión, el hallazgo más delicado. El paper (secciones 3.1, ambas versiones) afirma:

> "Se empleó un conjunto de datos recolectado de diversas fuentes de empleo tecnológico, incluyendo el Stack Overflow Developer Survey y plataformas de reclutamiento especializadas como LinkedIn Jobs y Glassdoor."

Evidencia en contra, revisando `tech_jobs_salaries.xlsx` directamente:
- Vocabulario cerrado y muy pequeño: solo **10 países** (USA, India, UK, Germany, France, Netherlands, Canada, Australia, Japan, Singapore), ~19 puestos de trabajo, ~17 skills — un diccionario fijo, típico de un generador sintético, no de datos agregados de encuestas/portales reales (que tendrían cientos de países y miles de títulos de puesto distintos en texto libre).
- **0% de missingness** en 200,000 filas — estadísticamente casi imposible en datos reales scrapeados de LinkedIn/Glassdoor o en una encuesta real como el Stack Overflow Developer Survey (que de hecho tiene enormes tasas de no-respuesta documentadas públicamente).
- La necesidad de "corregir" la moneda porque el dataset trae `currency` inconsistente con `country` (ver `country_currency_map` en el código) es un patrón típico de datasets sintéticos generados con ruido artificial introducido a propósito para practicar limpieza de datos (frecuente en datasets tipo Kaggle de práctica).
- No hay en el repo ningún script de scraping, ninguna API key, ningún archivo de licencia/atribución de Stack Overflow/LinkedIn/Glassdoor, ni URL de descarga real del dataset.

**Conclusión:** todo apunta a que `tech_jobs_salaries.xlsx` es un dataset sintético/simulado (del estilo de los que circulan en Kaggle para practicar regresión), y la afirmación de que proviene de Stack Overflow/LinkedIn/Glassdoor no está soportada por evidencia y probablemente es incorrecta. Esto ya está en el **paper publicado** (versión reducida), no solo en el borrador.

### 2.7 Referencias bibliográficas: al menos una tiene un DOI placeholder, y estaba en el paper ya publicado
- `[4] Zhu, X., et al. (2022). Tech Salary Prediction Using Machine Learning: A Comparative Study. Atlantis Press. https://doi.org/10.5551/j.sci.2022.xx` — el DOI termina literalmente en `.xx`, un patrón de placeholder no completado. Esta referencia aparece igual en **ambos** documentos, incluido el que se envió a publicar.
- La versión inicial también citaba `[19] Mills, K. T., et al. (2016). Global Disparities of Hypertension Prevalence...` — un paper de epidemiología/hipertensión sin ninguna relación con predicción salarial o ML. Fue correctamente eliminado en la versión reducida (buena señal de que hubo revisión), pero indica que el proceso de generación de referencias (posiblemente asistido por IA) coló al menos una cita fuera de tema, y quedó sin detectarse la de DOI placeholder.
- **Acción recomendada:** verificar una por una las ~19 referencias contra fuentes reales (existencia del DOI, autores, año) antes de reutilizar este texto en una tesis. Un comité revisa bibliografía con lupa, y una cita con DOI inventado es un problema serio de integridad académica si no se corrige.

---

## 3. Diferencia entre la versión inicial y la reducida (publicada)

No hay diferencias de contenido sustantivo — la reducida es una **compresión editorial** de la inicial, no una revisión metodológica:
- Se fusiona/resume prosa (mismos hallazgos, menos palabras).
- Se recorta la Tabla 4 (CV) de 9 modelos a 5 filas, y la Tabla 7 (SHAP top-20) a Tabla 6 (top-10).
- Se **elimina por completo** la subsección "4.3 Análisis de Estabilidad y Consistencia" (comparación de std entre folds, comentario sobre LightGBM/XGBoost siendo más inestables) — contenido analítico real que se perdió, no solo prosa redundante.
- Se corrige la lista de referencias quitando la cita de hipertensión fuera de tema.
- Los números clave (R²≈0.8075, RMSE≈29,748, MAPE≈26.72%, 66.7% de importancia de país) son **idénticos** entre ambas versiones — es decir, no hubo una nueva corrida de experimentos entre el borrador y el envío a publicación.

---

## 4. Qué necesitas resolver antes de escalar esto a tesis (checklist priorizado)

**Bloqueante / integridad académica (hacer primero, incluso si implica una fe de erratas del paper ya publicado):**
1. Verificar el origen real de `tech_jobs_salaries.xlsx`. Si es sintético, decirlo explícitamente en cualquier trabajo futuro y replantear qué se puede afirmar ("hallazgos sobre datos simulados" ≠ "hallazgos del mercado laboral tech real").
2. Auditar las ~19 referencias una por una (DOI real, autor real). Corregir o retirar la de Zhu et al. si no se puede verificar.
3. Decidir qué hacer con el paper ya publicado si el dataset resulta sintético y estaba presentado como real (posible corrección/aclaración con el coautor/revista, según corresponda).

**Rigor metodológico (necesario para que la tesis tenga una base sólida):**
4. Unificar en **un solo notebook reproducible** (partiendo de v4, que es el más completo) que corra de principio a fin sin pasos manuales, con semilla fija, y que exporte automáticamente TODOS los CSV/figuras que el texto cita — incluyendo Stacking, XGBoost, LightGBM, CatBoost en el mismo `results_test.csv`.
5. Reemplazar `feature_importances_` por SHAP real (`shap.TreeExplainer` para los modelos de árbol, `shap.LinearExplainer` o coeficientes para Ridge) si se va a seguir hablando de SHAP. Guardar los `shap_values` calculados, no solo el gráfico.
6. Si se mantiene el Stacking, documentar la fórmula real (con `passthrough=True`) o quitar el passthrough si se quiere que coincida con la fórmula simple del paper.
7. Fechar y justificar las tasas de cambio (o mejor: usar una API/tabla histórica real con fecha de referencia).
8. Reintroducir pruebas de significancia estadística (Wilcoxon, como en v3) sobre los resultados finales de v4, ya que el paper hace afirmaciones fuertes tipo "la diferencia no justifica la complejidad" sin soporte estadístico formal en la versión final.

---

## 5. Ideas de pivote para tesis (de más a menos recomendado)

**A. Interpretabilidad + equidad algorítmica real (mi recomendación principal).**
El paper promete SHAP, LIME, PDP y análisis de sesgo pero nunca los ejecuta de verdad. Ahí hay una tesis completa: construir el aparato real de XAI (SHAP + LIME + PDP/ICE) sobre el mejor modelo, y usarlo para sustentar (o refutar) la afirmación de que el país explica 66.7% — con SHAP real ese número casi seguro cambia. Combinarlo con mitigación de sesgos (reweighing, calibración post-hoc por grupo, adversarial debiasing) y medir el trade-off rendimiento/equidad antes-después. Es una contribución metodológica clara y defendible, y reutiliza casi todo el pipeline ya construido.

**B. Curaduría de datos reales como contribución central.**
Si el dataset actual es sintético, una tesis puede ganar mucho valor simplemente por reemplazarlo con datos reales: el Stack Overflow Developer Survey tiene datasets anuales públicos y reales descargables, y existen datasets reales de compensación tech (levels.fyi tiene datos públicos parciales, hay datasets de Glassdoor/Kaggle etiquetados como reales con su fuente clara). La contribución de tesis sería la curaduría/fusión de fuentes heterogéneas reales, con su propio tratamiento de missingness real (que el dataset actual no tiene por ser sintético). El pipeline de modelado actual serviría de baseline metodológico.

**C. Especialización geográfica + PPP, como ya sugiere el propio paper en "trabajos futuros".**
Modelar por región/país específico (para que la variable "país" deje de dominar todo) e incorporar paridad de poder adquisitivo. Es un pivote más acotado y menos arriesgado que A o B, pero también menos novedoso — el propio paper ya lo deja anotado como línea futura, así que un comité podría verlo como "hacer lo que ya dijiste que harías" en vez de una aportación nueva.

**D. Generalización temporal.**
Evaluar si el modelo entrenado en este dataset sigue siendo válido con datos más recientes (drift). Requiere conseguir una segunda muestra temporal, lo cual choca de nuevo con el problema de origen de datos (B).

Mi sugerencia concreta: **A primero, resolviendo honestamente el punto de origen de datos como prerrequisito** (aunque sea documentando "se usa un dataset sintético para el desarrollo metodológico, y se valida el hallazgo de dominancia geográfica con [dataset real X] como validación externa" — esto combina A+B sin duplicar el esfuerzo).

---

## 6. Sobre usar agentes/este software para lo que sigue

Para esta tarea (lectura, comparación de documentos, auditoría de código vs. texto) no hacía falta levantar un agente aparte: fue trabajo de inspección directa. Para las siguientes etapas, sí tiene sentido usar el patrón de este orquestador:

- **Dev**, con instrucciones exactas y acotadas, una vez decidas el pivote — por ejemplo: "en `salary_prediction_v4.ipynb`, reemplaza la celda 23 (placeholder de importancia) por un cálculo real de SHAP con `shap.TreeExplainer` sobre el pipeline de CatBoost ya entrenado en la celda 34, guarda los valores en `shap_values.npy` y genera `figures/fig_shap_summary.png`". Yo (Tech Lead) definiría cada tarea así de concreta, en lugar de pedirle a dev que "investigue cómo hacer SHAP".
- **QA**, después de cualquier cambio estructural al pipeline (p.ej. unificar v4, agregar SHAP real, quitar/ajustar el passthrough del stacking) — para validar que el notebook corre de punta a punta y que las cifras exportadas coinciden con las citadas en cualquier texto nuevo.
- Un agente de escritura académica/estilo APA dedicado **no está entre los agentes disponibles aquí** (solo dev y qa); si quieres ese tipo de apoyo para redactar la tesis en sí, sería una sesión/herramienta aparte, no algo que este orquestador de Dev/QA resuelva bien.

No creo que necesites un tercer tipo de agente técnico para esto: el cuello de botella no es de tooling, es de decisiones (qué pivote tomar, cómo resolver el origen del dataset) que te corresponden a ti antes de que valga la pena poner a un dev a picar código.
