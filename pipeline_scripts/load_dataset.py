import os
import pandas as pd


def load_data(**context):
    dataset_path = '/opt/airflow/data/Titanic-Dataset.csv'

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"File not found: {dataset_path}. "
            "Place Titanic-Dataset.csv in the airflow/data/ folder."
        )

    df = pd.read_csv(dataset_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}")

    context['ti'].xcom_push(key='dataset_path', value=dataset_path)