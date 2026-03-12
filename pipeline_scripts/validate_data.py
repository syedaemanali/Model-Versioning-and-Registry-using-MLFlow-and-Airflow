import pandas as pd


def validate_data(**context):
    ti = context['ti']
    dataset_path = ti.xcom_pull(task_ids='load_dataset', key='dataset_path')
    df = pd.read_csv(dataset_path)

    # fail on first attempt on purpose to show Airflow retry working
    if ti.try_number == 1:
        raise ValueError("Intentional failure on attempt 1. Airflow will retry.")

    total = len(df)
    age_pct = df['Age'].isnull().sum() / total * 100
    embarked_pct = df['Embarked'].isnull().sum() / total * 100

    print(f"Age missing: {age_pct:.2f}%")
    print(f"Embarked missing: {embarked_pct:.2f}%")

    if age_pct > 30:
        raise ValueError(f"Age missing {age_pct:.2f}% exceeds 30% limit.")
    if embarked_pct > 30:
        raise ValueError(f"Embarked missing {embarked_pct:.2f}% exceeds 30% limit.")

    print("Validation passed.")
    ti.xcom_push(key='dataset_path', value=dataset_path)