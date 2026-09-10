"""Explainability utilities using SHAP for model interpretation."""
import os
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline


def explain_model(pipe: Pipeline, X_train: pd.DataFrame, X_test: pd.DataFrame,
                  output_dir: str = 'results', sample_size: int = 100) -> dict:
    """
    Calculate SHAP values for the best trained model and generate summary plot.

    Parameters
    ----------
    pipe : Pipeline
        Fitted sklearn pipeline with preprocessor and model.
    X_train : pd.DataFrame
        Training features (used for background distribution).
    X_test : pd.DataFrame
        Test features for SHAP explanation.
    output_dir : str
        Directory to save outputs.
    sample_size : int
        Number of background samples for SHAP.

    Returns
    -------
    dict with top 5 features and SHAP summary path.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Extract preprocessor and model from pipeline
    preprocessor = pipe.named_steps['preprocessor']
    model = pipe.named_steps['model']

    # Transform data through preprocessor
    X_train_proc = preprocessor.transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # Get feature names after transformation
    feature_names = _get_feature_names(preprocessor, X_train)

    # Sample background for SHAP (KernelExplainer for broader compatibility)
    background = shap.sample(pd.DataFrame(X_train_proc, columns=feature_names),
                             min(sample_size, len(X_train_proc)))

    # Use KernelExplainer (works with any model)
    explainer = shap.KernelExplainer(model.predict, background)

    # Sample test data for efficiency
    test_sample = pd.DataFrame(X_test_proc, columns=feature_names)
    if len(test_sample) > 200:
        test_sample = test_sample.sample(200, random_state=42)

    # Calculate SHAP values
    shap_values = explainer.shap_values(test_sample)

    # Handle multi-output (take first output if array of arrays)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    # Get top 5 features by mean absolute SHAP value
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[-5:][::-1]
    top_features = [(feature_names[i], float(mean_abs_shap[i])) for i in top_indices]

    # Generate summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, test_sample, feature_names=feature_names,
                      show=False, max_display=10)
    plot_path = os.path.join(output_dir, 'shap_summary.png')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    return {
        'top_5_features': top_features,
        'shap_summary_path': plot_path,
        'mean_abs_shap': {feature_names[i]: float(mean_abs_shap[i])
                          for i in range(len(feature_names))}
    }


def _get_feature_names(preprocessor, X: pd.DataFrame) -> list:
    """Extract feature names from a fitted ColumnTransformer."""
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
