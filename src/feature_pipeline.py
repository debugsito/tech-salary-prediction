import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from src.encoders import TextEmbeddingTransformer, MeanTargetEncoder

NUM_FEATURES = ['years_experience','work_hours_per_week','job_satisfaction_score','company_rating','age','experience_num','education_num','company_size_num','remote_num','employment_num','exp_x_edu','exp_x_years','years_x_edu','productivity_score','hours_deviation','experience_per_age','is_high_demand','is_senior_plus','is_full_time','same_skills']
CAT_LOW = ['company_size','employment_type','experience_level','education_level','remote_type','gender','region']
CAT_HIGH = ['country']
TEXT_COLS = ['job_title','primary_skill','secondary_skill']

def load_processed(path='data/processed/tech_jobs_salaries_processed.parquet'):
    return pd.read_parquet(path)

def precompute_text_embeddings(df):
    # El embedding de cada fila es funcion determinista de su propio texto (no usa y ni otras filas), por eso es seguro calcularlo una sola vez antes del split train/test/CV sin fuga de informacion.
    transformer = TextEmbeddingTransformer()
    transformer.fit(df[TEXT_COLS])
    emb = transformer.transform(df[TEXT_COLS])
    emb_cols = [f'sbert_{i}' for i in range(emb.shape[1])]
    emb_df = pd.DataFrame(emb, columns=emb_cols, index=df.index)
    return pd.concat([df, emb_df], axis=1), emb_cols

def build_preprocessor(country_encoding, emb_cols):
    numeric_transformer = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    cat_low_transformer = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    if country_encoding == 'onehot':
        cat_high_transformer = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    elif country_encoding == 'target':
        cat_high_transformer = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('target', MeanTargetEncoder(smoothing=10.0))])
    else:
        raise ValueError(country_encoding)
    return ColumnTransformer([('num', numeric_transformer, NUM_FEATURES), ('cat_low', cat_low_transformer, CAT_LOW), ('cat_high', cat_high_transformer, CAT_HIGH), ('emb', 'passthrough', emb_cols)], remainder='drop')
