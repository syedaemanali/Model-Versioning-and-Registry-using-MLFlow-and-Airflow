import pandas as pd
import os


def encode_data(**context):
    ti = context['ti']
    missing_path = ti.xcom_pull(task_ids='handle_missing_values', key='missing_handled_path')
    features_path = ti.xcom_pull(task_ids='feature_engineering', key='features_path')

    df = pd.read_csv(missing_path)

    # Merge FamilySize and IsAlone from the parallel feature engineering task
    df_feat = pd.read_csv(features_path)[['PassengerId', 'FamilySize', 'IsAlone']]
    df = df.merge(df_feat, on='PassengerId', how='left')

    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

    df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], errors='ignore', inplace=True)

    out_dir = '/opt/airflow/data/processed'
    os.makedirs(out_dir, exist_ok=True)
    out_path = f'{out_dir}/titanic_encoded.csv'
    df.to_csv(out_path, index=False)

    print(f"Encoding done. Columns: {list(df.columns)}")
    ti.xcom_push(key='encoded_path', value=out_path)