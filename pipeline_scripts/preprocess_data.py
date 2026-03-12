import pandas as pd
import os


def preprocess_data(step: str, **context):
    ti = context['ti']
    dataset_path = (
        ti.xcom_pull(task_ids='validate_data', key='dataset_path') or
        ti.xcom_pull(task_ids='load_dataset', key='dataset_path')
    )
    df = pd.read_csv(dataset_path)
    out_dir = '/opt/airflow/data/processed'
    os.makedirs(out_dir, exist_ok=True)

    if step == 'missing':
        df['Age'].fillna(df['Age'].median(), inplace=True)
        df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
        df['Fare'].fillna(df['Fare'].median(), inplace=True)

        out_path = f'{out_dir}/titanic_missing_handled.csv'
        df.to_csv(out_path, index=False)
        print(f"Missing values handled. Saved to {out_path}")
        ti.xcom_push(key='missing_handled_path', value=out_path)

    elif step == 'features':
        # Create FamilySize and IsAlone features
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

        out_path = f'{out_dir}/titanic_features.csv'
        df.to_csv(out_path, index=False)
        print(f"Features created. Saved to {out_path}")
        ti.xcom_push(key='features_path', value=out_path)