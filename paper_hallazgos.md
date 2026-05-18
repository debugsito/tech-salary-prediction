# Predicción Salarial en Tecnología v4 — Hallazgos y análisis

## Resumen ejecutivo
Este estudio desarrolla un pipeline de predicción salarial con embeddings semánticos para títulos y habilidades, y compara modelos lineales y de boosting avanzados. Se empleó validación cruzada estratificada por región y evaluación en test con métricas robustas en escala log y USD. El mejor desempeño se obtuvo con un ensamble (Stacking) y un modelo lineal (Ridge) muy cercano, lo que sugiere que la señal dominante es consistente y casi lineal tras la normalización y el log-transform. El análisis de sesgos por género, región y educación identifica diferencias de error que deben reportarse como limitaciones y oportunidades de calibración.

## Datos
- **Tamaño del dataset:** 200,000 registros, 17 columnas originales.
- **Cobertura:** sin valores nulos en columnas clave (missingness $=0$).
- **Variables categóricas principales:** `job_title`, `company_size`, `employment_type`, `experience_level`, `education_level`, `country`, `currency`, `remote_type`, `primary_skill`, `secondary_skill`, `gender`.
- **Variables numéricas principales:** `years_experience`, `salary_local_currency`, `work_hours_per_week`, `job_satisfaction_score`, `company_rating`, `age`.

**Figura 0 (Missingness):** `figures/fig_missingness.png`.

## Preprocesamiento y feature engineering (paso a paso)
1. **Diagnóstico inicial**
  - Validación de tipos de datos y cobertura.
  - Justificación del uso de imputación mínima (sin nulos).

2. **Normalización del salario a USD**
  - Se corrige la moneda esperada por país.
  - Conversión a USD antes de cualquier filtrado para evitar sesgos de moneda.

3. **Tratamiento de outliers con IQR**
  - Se calcula $\text{IQR} = Q3 - Q1$ y se aplica clipping a $[Q1 - 1.5\cdot IQR,\, Q3 + 1.5\cdot IQR]$.
  - Esto reduce la influencia de colas extremas sin eliminar registros.

4. **Transformación del target**
  - Se aplica $\log(1 + \text{salary\_usd\_capped})$ para estabilizar varianza y mejorar ajuste de modelos lineales y boosting.

5. **Ingeniería de variables**
  - Mapas ordinales para experiencia, educación, tamaño de empresa y modalidad.
  - Interacciones: experiencia × educación, experiencia × años.
  - Variables de productividad: satisfacción × rating.
  - Variables binarias: alta demanda, seniority, full-time.
  - Región geográfica para reducir cardinalidad de país.

6. **Embeddings semánticos**
  - Se combinan `job_title`, `primary_skill`, `secondary_skill` y se codifican con `SentenceTransformer (all-MiniLM-L6-v2)`.
  - Esto preserva semántica de texto en lugar de usar encoding promedio por target.

**Figura 0b (Distribución target):** `figures/fig_target_distributions.png`.

## Modelos evaluados
- **Lineales:** Ridge (baseline fuerte con regularización L2).
- **Boosting:** HistGradientBoosting, XGBoost, LightGBM, CatBoost.
- **Ensamble:** Stacking (Ridge + HistGradientBoosting + CatBoost).

## Métricas
- **RMSE, MAE, MedAE** en escala log y USD.
- **MAPE** en USD.
- **$R^2$** en escala log.
- Se reportan métricas en USD para interpretación económica y en log para estabilidad estadística.

## Resultados de validación cruzada (5-fold, log-target)
| Modelo | RMSE_mean | RMSE_std | MAE_mean | MAE_std | R2_mean | R2_std |
|---|---:|---:|---:|---:|---:|---:|
| Ridge | 0.3071 | 0.0017 | 0.2535 | 0.0011 | 0.8052 | 0.0021 |
| HistGradientBoosting | 0.3077 | 0.0018 | 0.2538 | 0.0011 | 0.8044 | 0.0020 |
| CatBoost | 0.3083 | 0.0018 | 0.2542 | 0.0012 | 0.8037 | 0.0022 |
| XGBoost | 0.3096 | 0.0017 | 0.2548 | 0.0011 | 0.8020 | 0.0022 |
| LightGBM | 0.3103 | 0.0019 | 0.2551 | 0.0012 | 0.8011 | 0.0021 |

**Interpretación:** el desempeño es muy estable; Ridge lidera ligeramente. Esto sugiere que la relación salario–features, luego de ingeniería y log-transform, es casi lineal.

## Resultados en test (log-target y USD)
| Modelo | test_R2 | RMSE_USD | MAE_USD | MAPE (%) |
|---|---:|---:|---:|---:|
| Stacking | 0.8075 | 29,747.99 | 23,686.75 | 26.7165 |
| Ridge | 0.8075 | 29,757.90 | 23,688.98 | 26.7190 |
| HistGradientBoosting | 0.8070 | 29,773.36 | 23,719.50 | 26.7715 |
| CatBoost | 0.8063 | 29,792.30 | 23,728.20 | 26.7873 |
| XGBoost | 0.8050 | 29,872.34 | 23,780.36 | 26.8526 |
| LightGBM | 0.8042 | 29,898.94 | 23,798.76 | 26.8880 |

**Hallazgo clave:** El ensamble y Ridge son prácticamente equivalentes en desempeño; las mejoras de boosting avanzado son marginales, lo que respalda la hipótesis de una señal lineal dominante.

## Figuras generadas en el notebook (referencias para insertar en el paper)
- **Figura 0:** Missingness (top 10 variables). Archivo: `figures/fig_missingness.png`.
- **Figura 0b:** Distribución del salario (raw vs log1p) tras clipping. Archivo: `figures/fig_target_distributions.png`.
- **Figura 1:** Barras comparativas de RMSE/MAE/$R^2$/MAPE por modelo. Archivo: `figures/fig_model_metrics.png`.
- **Figura 2:** Real vs predicho (USD) + distribuciones de error (log/USD). Archivo: `figures/fig_error_calibration.png`.
- **Figura 3:** RMSE (log) por género. Archivo: `figures/fig_bias_gender.png`.
- **Figura 4:** RMSE (log) por región. Archivo: `figures/fig_bias_region.png`.
- **Figura 5:** RMSE (log) por nivel educativo. Archivo: `figures/fig_bias_education_level.png`.

## Análisis de sesgos
Se calcularon métricas por subgrupo para **género**, **región** y **nivel educativo** con el mejor modelo (Stacking/Ridge). Se reportan RMSE y MAE en escala log para comparar estabilidad. Se recomienda:
- Identificar grupos con mayor error relativo.
- Evaluar necesidad de variables adicionales (ej. costo de vida regional, seniority más granular).
- Considerar calibración posterior por grupo si los errores son sistemáticos.

## Discusión
1. **Embeddings semánticos:** aportan una representación más rica de títulos y skills; sin embargo, la señal global sigue dominada por variables estructurales (país, experiencia), lo que explica la ventaja competitiva de Ridge.
2. **Boosting avanzado:** mejora marginalmente, pero no supera el sesgo lineal del problema; el ensamble reduce RMSE de forma ligera.
3. **Sesgos:** se observan brechas por región/educación; esto sugiere limitaciones en el muestreo o factores exógenos no modelados.

## Conclusiones
- El pipeline v4 es **robusto, reproducible y con métricas estables**.
- El ensamble (Stacking) y Ridge son los modelos más consistentes.
- Los gráficos y el análisis de sesgos brindan un **valor científico** adecuado para presentación en paper.
- La generalización en test confirma que el desempeño no se debe a sobreajuste de CV.

## Sugerencias para el paper
- Incluir discusión sobre **dominancia de variables estructurales** (país/experiencia) vs embeddings.
- Incorporar análisis adicional de **regularización por país** o calibración por región.
- Reportar métricas por subgrupo como evidencia de equidad y limitaciones.
- Documentar parámetros de conversión monetaria con fecha de referencia.
