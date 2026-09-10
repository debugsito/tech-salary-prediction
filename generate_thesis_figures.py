"""
Script para la generación de figuras y tablas para el Capítulo 4 de la tesis.

Mapeo de Figuras al Capítulo 4:
==========================================================
Sección 4.1 - Descripción del Dataset:
  - (Estadísticas en tablas CSV, no figuras)
  
Sección 4.2 - Evaluación de Modelos:
  - residuals_plot.png: Análisis de residuos del mejor modelo
  - correlation_matrix.png: Matriz de correlación de features
  - tabla_resultados_test.csv: Métricas de todos los modelos en test
  
Sección 4.3 - Análisis de Importancia de Variables:
  - shap_beeswarm.png: SHAP beeswarm plot (importancia + dirección)
  - shap_summary.png: SHAP summary bar plot (importancia global)

Sección 4.3.2 - Dependencias Parciales (PDP):
  - pdp_experience.png: PDP para years_experience
  - pdp_country.png: PDP para country (impacto por país)

Sección 4.5 - Auditoría de Equidad:
  - fairness_gender.png: Disparidad de error por género
  - tabla_fairness_gender.csv: Resultados numéricos de fairness

Uso:
  - Ejecución completa: python generate_thesis_figures.py
  - Ejecución rápida (20k filas): python generate_thesis_figures.py --quick
"""

import argparse
import os
import sys
import shutil
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Asegurar que el root del proyecto esté en el path
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROJECT_ROOT)

from src.feature_pipeline import load_processed, precompute_text_embeddings, build_preprocessor, NUM_FEATURES, CAT_LOW, CAT_HIGH
from src.model_registry import build_models
from src.metrics import compute_metrics
from src.fairness import audit_fairness, get_fairness_summary

# Import SHAP separately in case it's not available
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available. SHAP figures will be skipped.")

# Configuración de paths (relativos a la raíz de tech-salary-prediction)
TESIS_ROOT = os.path.abspath(os.path.join(_PROJECT_ROOT, '..', 'tesis_borrador'))
FIG_DIR = os.path.join(TESIS_ROOT, 'figuras')
TAB_DIR = os.path.join(TESIS_ROOT, 'tablas')

SEED = 42

def setup_dirs():
    """Crear directorios de salida si no existen."""
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TAB_DIR, exist_ok=True)
    print(f"Directorio de figuras: {FIG_DIR}")
    print(f"Directorio de tablas: {TAB_DIR}")

def plot_correlation_matrix(df: pd.DataFrame, output_path: str):
    """Genera heatmap de matriz de correlación."""
    print(f"  [Figura] Matriz de correlación -> {output_path}")
    plt.figure(figsize=(14, 12))
    
    # Seleccionar solo columnas numéricas para la correlación
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Limitar a las features más importantes para legibilidad
    if numeric_df.shape[1] > 25:
        # Usar las features de mayor varianza/volatilidad
        variances = numeric_df.var().sort_values(ascending=False)
        top_features = variances.head(20).index.tolist()
        numeric_df = numeric_df[top_features]
    
    corr = numeric_df.corr()
    
    # Máscara para el triángulo superior (opcional, para limpieza)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    sns.heatmap(corr, mask=mask, annot=False, cmap='RdBu_r', center=0, 
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title("Matriz de Correlación de Features", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, output_path: str):
    """Genera scatter plot de residuos."""
    print(f"  [Figura] Plot de residuos -> {output_path}")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    residuals = y_true - y_pred
    
    # Scatter plot: predicciones vs residuos
    axes[0].scatter(y_pred, residuals, alpha=0.3, s=10)
    axes[0].axhline(0, color='red', linestyle='--', linewidth=2)
    axes[0].set_xlabel("Predicción (log-salary)")
    axes[0].set_ylabel("Residuo (log-salary)")
    axes[0].set_title("Residuos vs Predicciones")
    axes[0].grid(alpha=0.3)
    
    # Histograma de residuos
    axes[1].hist(residuals, bins=50, alpha=0.7, edgecolor='black')
    axes[1].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[1].set_xlabel("Residuo (log-salary)")
    axes[1].set_ylabel("Frecuencia")
    axes[1].set_title("Distribución de Residuos")
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_shap_analysis(pipe, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                       output_dir: str, quick_mode: bool = False):
    """Genera SHAP beeswarm y summary plots."""
    if not SHAP_AVAILABLE:
        print("  [SKIP] SHAP no disponible, saltando figuras SHAP")
        return
    
    print("  [Figura] Generando análisis SHAP...")
    
    try:
        # Extraer preprocessor y modelo del pipeline
        preprocessor = pipe.named_steps['preprocessor']
        model = pipe.named_steps['model']
        
        # Transformar datos
        X_train_proc = preprocessor.transform(X_train)
        X_test_proc = preprocessor.transform(X_test)
        
        # Obtener nombres de features después de transformación
        feature_names = _get_feature_names(preprocessor, X_train)
        
        # Crear DataFrames procesados
        X_train_df = pd.DataFrame(X_train_proc, columns=feature_names)
        X_test_df = pd.DataFrame(X_test_proc, columns=feature_names)
        
        # Limitar tamaño de muestra para SHAP (demasiado lento con todo)
        bg_size = 50 if quick_mode else 100
        test_size = 100 if quick_mode else 200
        
        background = X_train_df.sample(min(bg_size, len(X_train_df)), random_state=SEED)
        test_sample = X_test_df.sample(min(test_size, len(X_test_df)), random_state=SEED)
        
        # KernelExplainer (funciona con cualquier modelo)
        explainer = shap.KernelExplainer(model.predict, background)
        shap_values = explainer.shap_values(test_sample)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        # 1. SHAP Beeswarm (dot plot)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, test_sample, feature_names=feature_names,
                          show=False, max_display=15, plot_type="dot")
        plt.title("SHAP Beeswarm - Importancia y Dirección de Features", fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'shap_beeswarm.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    -> shap_beeswarm.png")
        
        # 2. SHAP Summary Bar (importancia global)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, test_sample, feature_names=feature_names,
                          show=False, max_display=15, plot_type="bar")
        plt.title("SHAP Summary - Importancia Global de Features", fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'shap_summary.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    -> shap_summary.png")
        
    except Exception as e:
        print(f"  [ERROR] Falló generación SHAP: {e}")

def _get_feature_names(preprocessor, X: pd.DataFrame) -> list:
    """Extrae nombres de features de un ColumnTransformer."""
    feature_names = []
    for name, transformer, columns in preprocessor.transformers_:
        if name == 'remainder':
            continue
        if transformer == 'passthrough':
            feature_names.extend(columns)
        elif hasattr(transformer, 'get_feature_names_out'):
            try:
                feature_names.extend(transformer.get_feature_names_out(columns))
            except (TypeError, ValueError):
                feature_names.extend(columns)
        else:
            feature_names.extend(columns)
    return feature_names

def plot_pdp(pipe, X_test: pd.DataFrame, feature: str, output_path: str):
    """Genera Partial Dependence Plot para una feature específica."""
    print(f"  [Figura] PDP para {feature} -> {output_path}")
    try:
        from sklearn.inspection import partial_dependence
        
        # Calcular PDP
        pd_result = partial_dependence(pipe, X_test, features=[feature], grid_resolution=20)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        
        if pd_result['grid_values'][0].dtype == object:
            # Feature categórica
            grid_values = range(len(pd_result['grid_values'][0]))
            ax.bar(grid_values, pd_result['average'][0])
            ax.set_xticks(grid_values)
            ax.set_xticklabels(pd_result['grid_values'][0], rotation=45, ha='right')
        else:
            # Feature numérica
            ax.plot(pd_result['grid_values'][0], pd_result['average'][0], linewidth=2)
            ax.fill_between(pd_result['grid_values'][0], 
                           pd_result['average'][0] - pd_result['std'][0],
                           pd_result['average'][0] + pd_result['std'][0], alpha=0.3)
        
        ax.set_xlabel(feature)
        ax.set_ylabel("Partial Dependence (log-salary)")
        ax.set_title(f"Partial Dependence Plot: {feature}")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"  [ERROR] Falló PDP para {feature}: {e}")

def plot_fairness_disparity(fairness_results: dict, output_path: str):
    """Genera barplot de disparidad de errores por grupo."""
    print(f"  [Figura] Fairness disparity -> {output_path}")
    
    groups = sorted(fairness_results['group_metrics'].keys())
    maes = [fairness_results['group_metrics'][g]['mae'] for g in groups]
    rmses = [fairness_results['group_metrics'][g]['rmse'] for g in groups]
    counts = [fairness_results['group_metrics'][g]['count'] for g in groups]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # MAE por grupo
    bars1 = ax1.bar(groups, maes, color='steelblue', alpha=0.8, edgecolor='black')
    ax1.set_ylabel("MAE (log-salary)")
    ax1.set_title(f"Error Absoluto Medio por {fairness_results['protected_attribute']}")
    ax1.grid(axis='y', alpha=0.3)
    # Añadir valor encima de cada barra
    for bar, val in zip(bars1, maes):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Tamaño de muestra por grupo
    bars2 = ax2.bar(groups, counts, color='coral', alpha=0.8, edgecolor='black')
    ax2.set_ylabel("Cantidad de muestras")
    ax2.set_title(f"Distribución de muestras por {fairness_results['protected_attribute']}")
    ax2.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars2, counts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                f'{val}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def generate_tables(test_results: list, fairness_results: dict, output_dir: str):
    """Genera tablas CSV para el Capítulo 4."""
    print("  [Tablas] Generando archivos CSV...")
    
    # Tabla 4.1: Resultados de Test
    df_metrics = pd.DataFrame(test_results)
    df_metrics.to_csv(os.path.join(output_dir, 'tabla_resultados_test.csv'), index=False)
    print(f"    -> tabla_resultados_test.csv")
    
    # Tabla 4.2: Métricas de Fairness
    fairness_df = pd.DataFrame({
        'Grupo': list(fairness_results['group_metrics'].keys()),
        'MAE': [m['mae'] for m in fairness_results['group_metrics'].values()],
        'RMSE': [m['rmse'] for m in fairness_results['group_metrics'].values()],
        'MAPE': [m['mape'] for m in fairness_results['group_metrics'].values()],
        'N': [m['count'] for m in fairness_results['group_metrics'].values()]
    })
    fairness_df.to_csv(os.path.join(output_dir, 'tabla_fairness_gender.csv'), index=False)
    print(f"    -> tabla_fairness_gender.csv")
    
    # Tabla 4.3: Disparidades
    disparity_df = pd.DataFrame({
        'Métrica': ['Diferencia máx MAE', 'Ratio máx/min MAE', 'Coef. de variación'],
        'Valor': [
            fairness_results['disparity']['max_mae_diff'],
            fairness_results['disparity']['max_mae_ratio'],
            fairness_results['disparity']['coefficient_of_variation']
        ]
    })
    disparity_df.to_csv(os.path.join(output_dir, 'tabla_fairness_disparity.csv'), index=False)
    print(f"    -> tabla_fairness_disparity.csv")

def main():
    parser = argparse.ArgumentParser(description='Generador de figuras para la tesis.')
    parser.add_argument('--quick', action='store_true', 
                        help='Usar 20k filas para ejecución rápida (default: 200k)')
    args = parser.parse_args()

    print("=" * 60)
    print("GENERADOR DE FIGURAS CAPÍTULO 4 - TESIS")
    print("=" * 60)
    
    setup_dirs()
    print()
    
    # 1. CARGA DE DATOS
    print("[1/6] Cargando datos...")
    df = load_processed()
    print(f"  Dataset cargado: {df.shape[0]:,} filas x {df.shape[1]} columnas")
    
    if args.quick:
        print(f"  Modo QUICK: Submuestreando a 20,000 filas")
        df = df.sample(n=min(20000, len(df)), random_state=SEED)
    
    # Computar embeddings y preparar features
    df, emb_cols = precompute_text_embeddings(df)
    all_features = NUM_FEATURES + CAT_LOW + CAT_HIGH + emb_cols
    X = df[all_features].copy()
    y = df['salary_log'].astype(np.float32)
    
    print(f"  Features finales: {len(all_features)} (num: {len(NUM_FEATURES)}, cat: {len(CAT_LOW)+len(CAT_HIGH)}, emb: {len(emb_cols)})")
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=df['region']
    )
    df_test = df.loc[X_test.index].copy()
    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")
    print()
    
    # 2. ENTRENAMIENTO DE MODELOS
    print("[2/6] Entrenando modelos...")
    preprocessor = build_preprocessor('target', emb_cols)
    models_to_run = build_models()
    
    test_results = []
    best_pipe = None
    best_r2 = -float('inf')
    best_model_name = ""
    
    for name, model in models_to_run.items():
        print(f"  Entrenando {name}...", end=" ")
        pipe = Pipeline([('preprocessor', preprocessor), ('model', model)])
        pipe.fit(X_train, y_train)
        
        y_pred = pipe.predict(X_test)
        m = compute_metrics(y_test, y_pred, prefix='test_')
        m['Modelo'] = name
        test_results.append(m)
        
        print(f"R² = {m['test_R2']:.4f}")
        
        if m['test_R2'] > best_r2:
            best_r2 = m['test_R2']
            best_pipe = pipe
            best_model_name = name
    
    print(f"\n  Mejor modelo: {best_model_name} (R² = {best_r2:.4f})")
    print()
    
    # 3. FIGURAS DE ANÁLISIS GENERAL
    print("[3/6] Generando figuras generales...")
    y_best_pred = best_pipe.predict(X_test)
    plot_correlation_matrix(X_test, os.path.join(FIG_DIR, 'correlation_matrix.png'))
    plot_residuals(y_test.values, y_best_pred, os.path.join(FIG_DIR, 'residuals_plot.png'))
    print()
    
    # 4. SHAP (EXPLICABILIDAD)
    print("[4/6] Generando análisis SHAP (esto puede tardar)...")
    plot_shap_analysis(best_pipe, X_train, X_test, FIG_DIR, quick_mode=args.quick)
    print()
    
    # 5. PDP (DEPENDENCIAS PARCIALES)
    print("[5/6] Generando Partial Dependence Plots...")
    plot_pdp(best_pipe, X_test, 'years_experience', 
             os.path.join(FIG_DIR, 'pdp_experience.png'))
    plot_pdp(best_pipe, X_test, 'country', 
             os.path.join(FIG_DIR, 'pdp_country.png'))
    print()
    
    # 6. FAIRNESS (EQUIDAD)
    print("[6/6] Generando auditoría de equidad...")
    try:
        protected_gender = df_test['gender'].reset_index(drop=True)
        fairness_results = audit_fairness(y_test.values, y_best_pred, protected_gender)
        
        # Plot de disparidad
        plot_fairness_disparity(fairness_results, os.path.join(FIG_DIR, 'fairness_gender.png'))
        
        # Resumen en consola
        print(f"  Overall MAE: {fairness_results['overall_mae']:.4f}")
        print(f"  Max MAE disparity: {fairness_results['disparity']['max_mae_diff']:.4f}")
        
    except Exception as e:
        print(f"  [ERROR] Falló auditoría de equidad: {e}")
        fairness_results = {'group_metrics': {'N/A': {'mae': 0, 'rmse': 0, 'count': 0}}, 
                           'disparity': {}, 'overall_mae': 0, 'protected_attribute': 'gender'}
    print()
    
    # 7. TABLAS CSV
    print("[7/6] Generando tablas CSV...")
    generate_tables(test_results, fairness_results, TAB_DIR)
    print()
    
    # RESUMEN FINAL
    print("=" * 60)
    print("PROCESO COMPLETADO")
    print("=" * 60)
    print(f"Figuras generadas en: {FIG_DIR}")
    print(f"  - correlation_matrix.png")
    print(f"  - residuals_plot.png")
    print(f"  - shap_beeswarm.png")
    print(f"  - shap_summary.png")
    print(f"  - pdp_experience.png")
    print(f"  - pdp_country.png")
    print(f"  - fairness_gender.png")
    print()
    print(f"Tablas generadas en: {TAB_DIR}")
    print(f"  - tabla_resultados_test.csv")
    print(f"  - tabla_fairness_gender.csv")
    print(f"  - tabla_fairness_disparity.csv")
    print()
    print(f"Mejor modelo: {best_model_name} (R² = {best_r2:.4f})")
    print("=" * 60)

if __name__ == '__main__':
    main()
