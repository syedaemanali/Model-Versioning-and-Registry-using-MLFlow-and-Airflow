import os
import mlflow
from mlflow.tracking import MlflowClient


def register_model(**context):
    ti = context['ti']
    run_id     = ti.xcom_pull(task_ids='evaluate_model', key='run_id')
    accuracy   = ti.xcom_pull(task_ids='evaluate_model', key='accuracy')
    model_type = ti.xcom_pull(task_ids='train_model', key='model_type')

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://host.docker.internal:5000"))
    client = MlflowClient()

    model_name = "TitanicSurvivalModel"
    result = mlflow.register_model(model_uri=f"runs:/{run_id}/model", name=model_name)

    client.update_registered_model(
        name=model_name,
        description=f"{model_type} model with accuracy {accuracy:.4f}"
    )
    print(f"Model registered: {model_name} v{result.version} | Accuracy: {accuracy:.4f}")


def reject_model(**context):
    ti = context['ti']
    accuracy = ti.xcom_pull(task_ids='evaluate_model', key='accuracy')
    run_id   = ti.xcom_pull(task_ids='evaluate_model', key='run_id')

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://host.docker.internal:5000"))
    client = MlflowClient()

    reason = f"Accuracy {accuracy:.4f} is below the 0.80 threshold."
    client.set_tag(run_id, "status", "rejected")
    client.set_tag(run_id, "rejection_reason", reason)
    print(f"Model rejected. {reason}")