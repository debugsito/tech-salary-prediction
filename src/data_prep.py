import os


def load_raw(path='tech_jobs_salaries.xlsx'):
    import pandas as pd
    return pd.read_excel(path)


def add_salary_usd(df):
    # Tasas fijas aproximadas, NO ligadas a una fecha historica real. TODO(usuario): reemplazar por tasas historicas con fecha de referencia antes de publicar resultados.
    import numpy as np
    work_df = df.copy()
    country_currency_map = {'USA': 'USD', 'India': 'INR', 'UK': 'GBP', 'Germany': 'EUR', 'France': 'EUR', 'Netherlands': 'EUR', 'Canada': 'CAD', 'Australia': 'AUD', 'Japan': 'JPY', 'Singapore': 'SGD'}
    currency_to_usd = {'USD': 1.0, 'INR': 0.012, 'EUR': 1.08, 'GBP': 1.27, 'CAD': 0.73, 'AUD': 0.66, 'JPY': 0.0067, 'SGD': 0.74}
    work_df['currency_corrected'] = work_df['country'].map(country_currency_map)
    work_df['currency_corrected'] = work_df['currency_corrected'].fillna(work_df.get('currency', 'USD'))
    work_df['currency_rate'] = work_df['currency_corrected'].map(currency_to_usd).fillna(1.0)
    work_df['salary_usd'] = work_df['salary_local_currency'] * work_df['currency_rate']
    q1 = work_df['salary_usd'].quantile(0.25); q3 = work_df['salary_usd'].quantile(0.75); iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr; upper_bound = q3 + 1.5 * iqr
    work_df['salary_usd_capped'] = work_df['salary_usd'].clip(lower_bound, upper_bound)
    work_df['salary_log'] = np.log1p(work_df['salary_usd_capped'])
    return work_df


def build_features(work_df):
    work_df = work_df.copy()
    exp_map = {'Entry': 0, 'Mid': 1, 'Senior': 2, 'Lead': 3}; edu_map = {'Self-taught': 0, 'Diploma': 1, 'Bachelor': 2, 'Master': 3, 'PhD': 4}
    size_map = {'Startup': 0, 'SME': 1, 'Mid-size': 2, 'Enterprise': 3}; remote_map = {'Onsite': 0, 'Hybrid': 1, 'Remote': 2}; emp_map = {'Freelance': 0, 'Contract': 1, 'Full-time': 2}
    work_df['experience_num'] = work_df['experience_level'].map(exp_map)
    work_df['education_num'] = work_df['education_level'].map(edu_map)
    work_df['company_size_num'] = work_df['company_size'].map(size_map)
    work_df['remote_num'] = work_df['remote_type'].map(remote_map)
    work_df['employment_num'] = work_df['employment_type'].map(emp_map)
    work_df['exp_x_edu'] = work_df['experience_num'] * work_df['education_num']
    work_df['exp_x_years'] = work_df['experience_num'] * work_df['years_experience']
    work_df['years_x_edu'] = work_df['years_experience'] * work_df['education_num']
    work_df['productivity_score'] = work_df['job_satisfaction_score'] * work_df['company_rating']
    work_df['hours_deviation'] = work_df['work_hours_per_week'] - 40
    work_df['experience_per_age'] = work_df['years_experience'] / work_df['age']
    high_demand_roles = ['data scientist', 'ml engineer', 'ai researcher', 'data engineer', 'software engineer']
    work_df['is_high_demand'] = work_df['job_title'].str.lower().fillna('').apply(lambda x: int(any(r in x for r in high_demand_roles)))
    work_df['is_senior_plus'] = (work_df['experience_level'].isin(['Senior', 'Lead'])).astype(int)
    work_df['is_full_time'] = (work_df['employment_type'] == 'Full-time').astype(int)
    work_df['same_skills'] = (work_df['primary_skill'] == work_df['secondary_skill']).astype(int)
    region_map = {'USA': 'North America', 'Canada': 'North America', 'UK': 'Europe', 'Germany': 'Europe', 'France': 'Europe', 'Netherlands': 'Europe', 'India': 'Asia', 'Japan': 'Asia', 'Singapore': 'Asia', 'Australia': 'Oceania'}
    work_df['region'] = work_df['country'].map(region_map).fillna('Other')
    return work_df


if __name__ == '__main__':
    df = load_raw()
    df = add_salary_usd(df)
    df = build_features(df)

    print(df.shape)
    print(df[['salary_usd', 'salary_log']].describe())

    out_dir = os.path.join('data', 'processed')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'tech_jobs_salaries_processed.parquet')
    try:
        df.to_parquet(out_path)
        print(f'Guardado en {out_path}')
    except Exception as e:
        csv_path = os.path.join(out_dir, 'tech_jobs_salaries_processed.csv')
        df.to_csv(csv_path, index=False)
        print(f'No se pudo guardar como parquet ({e}); guardado como CSV en {csv_path}')
