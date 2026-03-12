from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.trigger_rule import TriggerRule
from datetime import datetime, timedelta
import sys

sys.path.insert(0, '/opt/airflow')

from pipeline_scripts.validate_data import validate_data
from pipeline_scripts.preprocess_data import preprocess_data
from pipeline_scripts.encode_data import encode_data
from pipeline_scripts.train_model import train_model
from pipeline_scripts.evaluate_model import evaluate_model
from pipeline_scripts.branching import branching_logic
from pipeline_scripts.register_model import register_model, reject_model
from pipeline_scripts.load_dataset import load_data



default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(seconds=10),
}

with DAG(
    dag_id="titanic_ml_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
    default_args=default_args,
    tags=["titanic", "mlflow"],
) as dag:

    start = BashOperator(
        task_id="start_pipeline",
        bash_command="echo 'Pipeline Started'",
    )

    data_ingestion = PythonOperator(
        task_id="load_dataset",
        python_callable=load_data,
    )

    data_validation = PythonOperator(
        task_id="validate_data",
        python_callable=validate_data,
        retries=2,
        retry_delay=timedelta(seconds=5),
    )

    # handle_missing and feature_engineering run in parallel
    handle_missing = PythonOperator(
        task_id="handle_missing_values",
        python_callable=preprocess_data,
        op_kwargs={"step": "missing"},
    )

    feature_engineering = PythonOperator(
        task_id="feature_engineering",
        python_callable=preprocess_data,
        op_kwargs={"step": "features"},
    )

    encoding = PythonOperator(
        task_id="encode_data",
        python_callable=encode_data,
    )

    model_training = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
    )

    model_evaluation = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_model,
    )

    branch = BranchPythonOperator(
        task_id="branch_on_accuracy",
        python_callable=branching_logic,
    )

    register = PythonOperator(
        task_id="register_model",
        python_callable=register_model,
    )

    reject = PythonOperator(
        task_id="reject_model",
        python_callable=reject_model,
    )

    # end runs after either register or reject succeeds
    end = BashOperator(
        task_id="end_pipeline",
        bash_command="echo 'Pipeline Finished'",
        trigger_rule=TriggerRule.ONE_SUCCESS,
    )

    start >> data_ingestion >> data_validation
    data_validation >> [handle_missing, feature_engineering]
    [handle_missing, feature_engineering] >> encoding
    encoding >> model_training >> model_evaluation >> branch
    branch >> [register, reject]
    [register, reject] >> end