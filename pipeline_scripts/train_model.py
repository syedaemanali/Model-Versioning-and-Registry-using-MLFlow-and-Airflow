import os
import pickle
import logging
import tempfile
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

LOG_DIR = "/opt/airflow/logs/pipeline"

def get_logger(task_name: str) -> logging.Logger:
    os.makedirs(LOG_DIR, exist_ok=True)
    logger = logging.getLogger(task_name)
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                                datefmt="%Y-%m-%d %H:%M:%S")
        fh = logging.FileHandler(os.path.join(LOG_DIR, f"{task_name}.log"))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger


def train_model(**context):
    log = get_logger("train_model")
    log.info("Task: train_model — started")

    ti = context['ti']
    encoded_path = ti.xcom_pull(task_ids='encode_data', key='encoded_path')
    log.info(f"Encoded dataset path pulled from XCom: {encoded_path}")

    try:
        df = pd.read_csv(encoded_path)
        log.info(f"Dataset loaded — shape: {df.shape}")
    except Exception as e:
        log.error(f"Failed to load dataset: {e}", exc_info=True)
        raise

    X = df.drop('Survived', axis=1)
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    log.info(f"Train size: {len(X_train)}  |  Test size: {len(X_test)}")

    try:
        from airflow.models import Variable
        model_type   = Variable.get("model_type",    default_var="RandomForest")
        n_estimators = int(Variable.get("n_estimators", default_var=100))
        max_depth    = Variable.get("max_depth", default_var="None")
        max_depth    = None if max_depth == "None" else int(max_depth)
        C_val        = float(Variable.get("C", default_var=1.0))
        max_iter     = int(Variable.get("max_iter", default_var=200))
        log.info(f"Airflow Variables read — model_type={model_type}, "
                 f"n_estimators={n_estimators}, max_depth={max_depth}, "
                 f"C={C_val}, max_iter={max_iter}")
    except Exception as e:
        log.warning(f"Could not read Airflow Variables ({e}) — using defaults")
        model_type, n_estimators, max_depth, C_val, max_iter = "RandomForest", 100, None, 1.0, 200

    tracking_uri = "http://mlflow:5000"
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    os.environ["MLFLOW_ARTIFACTS_URI"] = tracking_uri
    os.environ["GIT_PYTHON_REFRESH"] = "quiet"
    mlflow.set_tracking_uri(tracking_uri)
    log.info(f"MLflow tracking URI: {tracking_uri}")

    try:
        mlflow.set_experiment("Titanic_Survival_Prediction")

        with mlflow.start_run() as run:
            mlflow.log_param("model_type",   model_type)
            mlflow.log_param("dataset_size", len(df))

            if model_type == "RandomForest":
                mlflow.log_param("n_estimators", n_estimators)
                mlflow.log_param("max_depth",    max_depth)
                model = RandomForestClassifier(n_estimators=n_estimators,
                                               max_depth=max_depth,
                                               random_state=42)
                log.info(f"Training RandomForest — n_estimators={n_estimators}, max_depth={max_depth}")
            else:
                mlflow.log_param("C",        C_val)
                mlflow.log_param("max_iter", max_iter)
                model = LogisticRegression(C=C_val, max_iter=max_iter, random_state=42)
                log.info(f"Training LogisticRegression — C={C_val}, max_iter={max_iter}")

            model.fit(X_train, y_train)
            log.info("Model training complete")

            # Save to temp dir first, then upload via HTTP to mlflow server
            with tempfile.TemporaryDirectory() as tmp_dir:
                model_tmp_path = os.path.join(tmp_dir, "model")
                mlflow.sklearn.save_model(model, model_tmp_path)
                mlflow.log_artifacts(model_tmp_path, artifact_path="model")
                log.info("Model artifact uploaded to MLflow via HTTP")

            run_id = run.info.run_id
            log.info(f"MLflow run_id: {run_id}")

    except Exception as e:
        log.error(f"MLflow / training failed: {e}", exc_info=True)
        raise

    # Save model pickle locally for downstream tasks
    model_path = '/opt/airflow/data/processed/model.pkl'
    try:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        log.info(f"Model saved to: {model_path}")
    except Exception as e:
        log.error(f"Failed to save model pickle: {e}", exc_info=True)
        raise

    ti.xcom_push(key='run_id',       value=run_id)
    ti.xcom_push(key='model_path',   value=model_path)
    ti.xcom_push(key='encoded_path', value=encoded_path)
    ti.xcom_push(key='model_type',   value=model_type)
    log.info("XCom values pushed successfully")
    log.info(f"Task: train_model — finished  |  log saved to {LOG_DIR}/train_model.log")
