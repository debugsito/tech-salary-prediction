"""
Script para generar todas las figuras del Capítulo 4 de la tesis.
- Usa solo 10k filas, Ridge y HGB, 3-fold CV
- Genera 7 figuras PNG + 2 tablas CSV
- Salida: ../tesis_borrador/capitulo4/figuras/ y tablas/
"""

import sys
import os
import glob

# Allow direct execution: python src/generate_thesis_figures.py
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import partial_dependence, PartialDependenceDisplay

from src.feature_pipeline import load_processed, precompute_text_embeddings, build_preprocessor, NUM_FEATURES, CAT_LOW, CAT_HIGH
from src.metrics import compute_metrics
from src.explainability import explain_model
from src.fairness import audit_fairness

SEED = 42
MAX_ROWS = 10000
CV_FOLDS = 3

# Rutas de salida (relativas al script)
OUTPUT_DIR_FIGURAS = '../tesis_borrador/capitulo4/figuras/'
OUTPUT_DIR_TABLAS = '../tesis_borrador/capitulo4/tablas/'


def clean_old_figures(fig_dir, tab_dir):
    """Elimina archivos PNG y CSV antiguos antes de generar nuevos."""
    for pattern in ['*.png', '*.csv']:
        for f in glob.glob(os.path.join(fig_dir, pattern)):
            os.remove(f)
        for f in glob.glob(os.path.join(tab_dir, pattern)):
            os.remove(f)
    print(f"Carpetas limpiadas: {fig_dir}, {tab_dir}")


def train_models_quick(X_train, y_train, X_test, y_test, preprocessor):
    """Entrena Ridge y HGB con 3-fold CV, retorna métricas y modelos entrenados."""
    models = {
        'Ridge': Ridge(alpha=1.0),
        'HistGradientBoosting': HistGradientBoostingRegressor(
            max_depth=10, learning_rate=0.1, max_iter=300, random_state=SEED
        )
    }
    
    kfold = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    scoring = {'rmse': 'neg_root_mean_squared_error', 'mae': 'neg_mean_absolute_error', 'r2': 'r2'}
    
    cv_rows, test_rows = [], []
    trained_pipes = {}
    
    for name, model in models.items():
        print(f"  Entrenando {name}...")
        pipe = Pipeline([('preprocessor', preprocessor), ('model', model)])
        
        # Cross-validation
        cv_res = cross_validate(pipe, X_train, y_train, cv=kfold, scoring=scoring, n_jobs=2)
        rmse = -cv_res['test_rmse']
        mae = -cv_res['test_mae']
        r2 = cv_res['test_r2']
        
        cv_rows.append({
            'Modelo': name,
            'RMSE_mean': rmse.mean(),
            'RMSE_std': rmse.std(),
            'MAE_mean': mae.mean(),
            'MAE_std': mae.std(),
            'R2_mean': r2.mean(),
            'R2_std': r2.std(),
            'fold1_r2': r2[0],
            'fold2_r2': r2[1],
            'fold3_r2': r2[2]
        })
        
        # Entrenar en full train y evaluar en test
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        m = compute_metrics(y_test, y_pred, prefix='test_')
        m['Modelo'] = name
        test_rows.append(m)
        trained_pipes[name] = pipe
        
        print(f"    CV R²: {r2.mean():.4f} (±{r2.std():.4f}) | Test R²: {m.get('test_R2', float('nan')):.4f}")
    
    return cv_rows, test_rows, trained_pipes


def plot_salary_distributions(df, output_path):
    """Figura 4.1: Histogramas comparativos salary_usd vs salary_log."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Salary USD original
    salary_usd = np.expm1(df['salary_log'])
    axes[0].hist(salary_usd, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    axes[0].set_xlabel('Salary (USD)', fontsize=11)
    axes[0].set_ylabel('Frecuencia', fontsize=11)
    axes[0].set_title('(a) Distribución de Salarios (USD)', fontsize=12, fontweight='bold')
    axes[0].axvline(salary_usd.median(), color='red', linestyle='--', label=f'Mediana: ${salary_usd.median():.0f}')
    axes[0].legend()
    
    # Salary log
    axes[1].hist(df['salary_log'], bins=50, color='forestgreen', edgecolor='white', alpha=0.8)
    axes[1].set_xlabel('log(Salary + 1)', fontsize=11)
    axes[1].set_ylabel('Frecuencia', fontsize=11)
    axes[1].set_title('(b) Distribución de Salarios Transformada (log)', fontsize=12, fontweight='bold')
    axes[1].axvline(df['salary_log'].median(), color='red', linestyle='--', label=f'Mediana: {df["salary_log"].median():.3f}')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {output_path}")


def plot_shap_analysis(pipe, X_train, X_test, output_dir):
    """Figuras 4.2 y 4.3: SHAP beeswarm y summary bar."""
    try:
        preprocessor = pipe.named_steps['preprocessor']
        model = pipe.named_steps['model']
        
        X_train_proc = preprocessor.transform(X_train)
        X_test_proc = preprocessor.transform(X_test)
        
        # Feature names
        feature_names = []
        for name, transformer, columns in preprocessor.transformers_:
            if name == 'remainder':
                continue
            if transformer == 'passthrough':
                feature_names.extend(columns)
            elif hasattr(transformer, 'get_feature_names_out'):
                try:
                    feature_names.extend(transformer.get_feature_names_out(columns))
                except:
                    feature_names.extend(columns)
            else:
                feature_names.extend(columns)
        
        # Muestrear background y test
        background = shap.sample(pd.DataFrame(X_train_proc, columns=feature_names), min(100, len(X_train_proc)))
        test_sample = pd.DataFrame(X_test_proc, columns=feature_names)
        if len(test_sample) > 200:
            test_sample = test_sample.sample(200, random_state=42)
        
        explainer = shap.KernelExplainer(model.predict, background)
        shap_values = explainer.shap_values(test_sample)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        # Figura 4.2: Beeswarm plot
        plt.figure(figsize=(10, 7))
        shap.summary_plot(shap_values, test_sample, feature_names=feature_names, show=False, max_display=15, plot_size=None)
        plt.tight_layout()
        path_4_2 = os.path.join(output_dir, 'figura_4_2_shap_beeswarm.png')
        plt.savefig(path_4_2, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Guardado: {path_4_2}")
        
        # Figura 4.3: Summary bar
        plt.figure(figsize=(10, 7))
        shap.summary_plot(shap_values, test_sample, feature_names=feature_names, show=False, max_display=15, plot_type='bar')
        plt.tight_layout()
        path_4_3 = os.path.join(output_dir, 'figura_4_3_shap_summary.png')
        plt.savefig(path_4_3, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Guardado: {path_4_3}")
        
        return True
    except Exception as e:
        print(f"  Advertencia: Error en SHAP analysis: {e}")
        return False


def plot_pdp_experience(pipe, X_train, output_path):
    """Figura 4.4: PDP de years_experience."""
    try:
        features_to_plot = ['years_experience']
        feature_names = list(X_train.columns)
        
        display = PartialDependenceDisplay.from_estimator(
            pipe, X_train, features_to_plot, feature_names=feature_names,
            kind='average', grid_resolution=20, n_jobs=1
        )
        display.figure_.suptitle('Partial Dependence: Years of Experience', fontsize=12, fontweight='bold')
        display.figure_.set_size_inches(8, 5)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Guardado: {output_path}")
        return True
    except Exception as e:
        print(f"  Advertencia: Error en PDP experience: {e}")
        # Crear figura de fallback
        plt.figure(figsize=(8, 5))
        plt.text(0.5, 0.5, f'PDP no disponible\n{str(e)[:100]}', 
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)
        plt.title('Partial Dependence: Years of Experience (Error)', fontsize=12)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return False


def plot_pdp_country(pipe, X_train, output_path):
    """Figura 4.5: PDP de country (top 10 países)."""
    try:
        if 'country' not in X_train.columns:
            print("  Advertencia: 'country' no está en features")
            return False
        
        # Top 10 países
        top_countries = X_train['country'].value_counts().head(10).index.tolist()
        mask = X_train['country'].isin(top_countries)
        X_sample = X_train[mask].sample(min(500, mask.sum()), random_state=42)
        
        # Para categorías, hacemos un plot manual
        predictions_by_country = []
        for country in top_countries:
            X_temp = X_sample.copy()
            X_temp['country'] = country
            preds = pipe.predict(X_temp)
            predictions_by_country.append(preds.mean())
        
        plt.figure(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0, 0.8, len(top_countries)))
        bars = plt.barh(top_countries[::-1], predictions_by_country[::-1], color=colors[::-1])
        plt.xlabel('Predicción promedio (log-salary)', fontsize=11)
        plt.ylabel('País', fontsize=11)
        plt.title('Partial Dependence: Country (Top 10)', fontsize=12, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Guardado: {output_path}")
        return True
    except Exception as e:
        print(f"  Advertencia: Error en PDP country: {e}")
        return False


def plot_cv_comparison(cv_df, output_path):
    """Figura 4.6: Boxplot comparando Ridge vs HGB en CV."""
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Preparar datos para boxplot
        models = cv_df['Modelo'].values
        data_r2 = []
        labels = []
        
        for _, row in cv_df.iterrows():
            r2_folds = [row['fold1_r2'], row['fold2_r2'], row['fold3_r2']]
            data_r2.append(r2_folds)
            labels.append(row['Modelo'])
        
        bp = ax.boxplot(data_r2, labels=labels, patch_artist=True, showmeans=True)
        colors = ['lightcoral', 'lightblue']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_ylabel('R² Score', fontsize=11)
        ax.set_xlabel('Modelo', fontsize=11)
        ax.set_title('Comparación de Modelos en Validación Cruzada (3-fold)', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1)
        
        # Añadir valores medios
        for i, row in cv_df.iterrows():
            ax.text(i+1, row['R2_mean'] + 0.02, f"R²={row['R2_mean']:.3f}", 
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Guardado: {output_path}")
        return True
    except Exception as e:
        print(f"  Advertencia: Error en CV comparison: {e}")
        return False


def plot_fairness_by_gender(pipe, X_test, y_test, df_test, output_path):
    """Figura 4.7: Barras de MAE por género."""
    try:
        y_pred = pipe.predict(X_test)
        
        if 'gender' not in df_test.columns:
            print("  Advertencia: 'gender' no está en df_test")
            return False
        
        protected = df_test['gender'].reset_index(drop=True)
        audit_result = audit_fairness(y_test.values, y_pred, protected)
        
        groups = list(audit_result['group_metrics'].keys())
        maes = [audit_result['group_metrics'][g]['mae'] for g in groups]
        counts = [audit_result['group_metrics'][g]['count'] for g in groups]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.Set2(np.linspace(0, 1, len(groups)))
        bars = ax.bar(groups, maes, color=colors, edgecolor='black', linewidth=0.5)
        
        # Añadir valores sobre barras
        for bar, mae, count in zip(bars, maes, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                   f'{mae:.3f}\n(n={count})', ha='center', va='bottom', fontsize=9)
        
        ax.set_ylabel('Mean Absolute Error (MAE)', fontsize=11)
        ax.set_xlabel('Género', fontsize=11)
        ax.set_title('Fairness Audit: MAE por Género', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Añadir estadísticas de disparidad
        disparity = audit_result['disparity']
        ax.text(0.5, 0.95, f"Max MAE Diff: {disparity['max_mae_diff']:.4f}\nOverall MAE: {audit_result['overall_mae']:.4f}",
               transform=ax.transAxes, ha='center', va='top', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Guardado: {output_path}")
        return True
    except Exception as e:
        print(f"  Advertencia: Error en fairness plot: {e}")
        return False


def main():
    """Función principal que genera todas las figuras y tablas del Capítulo 4."""
    generated_files = []
    
    # Crear directorios si no existen
    os.makedirs(OUTPUT_DIR_FIGURAS, exist_ok=True)
    os.makedirs(OUTPUT_DIR_TABLAS, exist_ok=True)
    
    print("=" * 60)
    print("GENERACIÓN DE FIGURAS CAPÍTULO 4 - TESIS")
    print("=" * 60)
    
    # Limpiar figuras antiguas
    clean_old_figures(OUTPUT_DIR_FIGURAS, OUTPUT_DIR_TABLAS)
    print()
    
    # Cargar datos
    print("Cargando datos...")
    df = load_processed()
    print(f"  Dataset cargado: {len(df)} filas totales")
    
    # Samplear a 10k filas
    df = df.sample(n=min(MAX_ROWS, len(df)), random_state=SEED).reset_index(drop=True)
    print(f"  Usando {len(df)} filas (muestreo aleatorio)")
    
    # Precomputar embeddings
    print("Precomputando embeddings...")
    df, emb_cols = precompute_text_embeddings(df)
    
    # Preparar features
    all_features = NUM_FEATURES + CAT_LOW + CAT_HIGH + emb_cols
    X = df[all_features].copy()
    y = df['salary_log'].astype(np.float32)
    
    # Split
    print("Dividiendo datos (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=df['region']
    )
    df_test = df.loc[X_test.index].copy()
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")
    print()
    
    # Preprocesador con target encoding para country
    preprocessor = build_preprocessor('target', emb_cols)
    
    # ============================================
    # 1. Figura 4.1: Distribución de salarios
    # ============================================
    print("Generando Figura 4.1: Distribución de salarios...")
    path_4_1 = os.path.join(OUTPUT_DIR_FIGURAS, 'figura_4_1_dist_salary.png')
    plot_salary_distributions(df, path_4_1)
    generated_files.append(path_4_1)
    print()
    
    # ============================================
    # Entrenar modelos
    # ============================================
    print("Entrenando modelos (Ridge + HGB, 3-fold CV)...")
    cv_rows, test_rows, trained_pipes = train_models_quick(
        X_train, y_train, X_test, y_test, preprocessor
    )
    cv_df = pd.DataFrame(cv_rows)
    test_df = pd.DataFrame(test_rows)
    print()
    
    # ============================================
    # Guardar tablas 4.2 y 4.3
    # ============================================
    print("Guardando tablas...")
    
    # Tabla 4.2: CV Results
    path_cv_table = os.path.join(OUTPUT_DIR_TABLAS, 'tabla_4_2_cv_results.csv')
    cv_df[['Modelo', 'R2_mean', 'R2_std', 'RMSE_mean', 'RMSE_std', 
           'MAE_mean', 'MAE_std']].to_csv(path_cv_table, index=False, float_format='%.6f')
    print(f"  Guardado: {path_cv_table}")
    generated_files.append(path_cv_table)
    
    # Tabla 4.3: Test Results
    path_test_table = os.path.join(OUTPUT_DIR_TABLAS, 'tabla_4_3_test_results.csv')
    test_cols = [c for c in test_df.columns if c.startswith('test_') or c == 'Modelo']
    test_df[test_cols].to_csv(path_test_table, index=False, float_format='%.6f')
    print(f"  Guardado: {path_test_table}")
    generated_files.append(path_test_table)
    print()
    
    # Modelo ganador para análisis posteriores (mejor R2 en test)
    best_model_name = test_df.loc[test_df['test_R2'].idxmax(), 'Modelo']
    best_pipe = trained_pipes[best_model_name]
    print(f"Mejor modelo seleccionado para SHAP/fairness: {best_model_name} (test R²={test_df['test_R2'].max():.4f})")
    print()
    
    # ============================================
    # 2-3. Figuras SHAP
    # ============================================
    print("Generando Figuras 4.2 y 4.3: SHAP analysis...")
    plot_shap_analysis(best_pipe, X_train, X_test, OUTPUT_DIR_FIGURAS)
    generated_files.append(os.path.join(OUTPUT_DIR_FIGURAS, 'figura_4_2_shap_beeswarm.png'))
    generated_files.append(os.path.join(OUTPUT_DIR_FIGURAS, 'figura_4_3_shap_summary.png'))
    print()
    
    # ============================================
    # 4. Figura 4.4: PDP Experience
    # ============================================
    print("Generando Figura 4.4: PDP Experience...")
    path_4_4 = os.path.join(OUTPUT_DIR_FIGURAS, 'figura_4_4_pdp_experience.png')
    plot_pdp_experience(best_pipe, X_train.head(500), path_4_4)
    generated_files.append(path_4_4)
    print()
    
    # ============================================
    # 5. Figura 4.5: PDP Country
    # ============================================
    print("Generando Figura 4.5: PDP Country...")
    path_4_5 = os.path.join(OUTPUT_DIR_FIGURAS, 'figura_4_5_pdp_country.png')
    plot_pdp_country(best_pipe, X_train, path_4_5)
    generated_files.append(path_4_5)
    print()
    
    # ============================================
    # 6. Figura 4.6: CV Comparison
    # ============================================
    print("Generando Figura 4.6: CV Comparison...")
    path_4_6 = os.path.join(OUTPUT_DIR_FIGURAS, 'figura_4_6_cv_comparison.png')
    plot_cv_comparison(cv_df, path_4_6)
    generated_files.append(path_4_6)
    print()
    
    # ============================================
    # 7. Figura 4.7: Fairness by Gender
    # ============================================
    print("Generando Figura 4.7: Fairness by Gender...")
    path_4_7 = os.path.join(OUTPUT_DIR_FIGURAS, 'figura_4_7_fairness_by_gender.png')
    plot_fairness_by_gender(best_pipe, X_test, y_test, df_test, path_4_7)
    generated_files.append(path_4_7)
    print()
    
    # ============================================
    # Resumen final
    # ============================================
    print("=" * 60)
    print("GENERACIÓN COMPLETADA")
    print("=" * 60)
    print(f"\nArchivos generados ({len(generated_files)}):")
    for f in generated_files:
        exists = "✓" if os.path.exists(f) else "✗"
        size = os.path.getsize(f) if os.path.exists(f) else 0
        print(f"  {exists} {f} ({size/1024:.1f} KB)")
    
    return generated_files


if __name__ == '__main__':
    main()
