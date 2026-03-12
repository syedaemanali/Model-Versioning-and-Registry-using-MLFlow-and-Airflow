# i221936 Assignment 2 DSB — Titanic ML Pipeline

End-to-end machine learning pipeline using Apache Airflow for orchestration and MLflow for experiment tracking. Predicts Titanic passenger survival using a DAG with parallel tasks, branching logic, and model registry.

---

## Requirements

- Docker Desktop (running)
- Python 3.10+
- 8GB RAM recommended

---

## STEP 1 — Create the .env file

Open PowerShell in the `airflow` folder and run:

```powershell
[System.IO.File]::WriteAllLines("$PWD\.env", @("AIRFLOW_UID=50000"))
```

This must be UTF-8 encoded or Docker will complain.

---

## STEP 2 — Start Everything

```powershell
cd i221936_Assignment2_DSB\airflow
docker compose up -d
```

Wait about 2 minutes for all containers to become healthy, then check:

```powershell
docker compose ps
```

You should see all services as `healthy`:
- `airflow-webserver` — Airflow UI at **http://localhost:8080**
- `airflow-scheduler`
- `airflow-worker`
- `airflow-triggerer`
- `airflow-mlflow-1` — MLflow UI at **http://localhost:5000**
- `postgres`
- `redis`

Default Airflow login: `airflow / airflow`

---

## STEP 3 — Verify DAG Loads

```powershell
docker exec airflow-airflow-webserver-1 airflow dags list-import-errors
docker exec airflow-airflow-webserver-1 airflow dags list
docker exec airflow-airflow-webserver-1 airflow dags unpause titanic_ml_pipeline
```

---

## STEP 4 — Run the 3 Experiments

Wait for each run to fully complete in the Airflow UI before triggering the next one.

**Experiment 1 — Random Forest (shallow)**
```powershell
docker exec airflow-airflow-webserver-1 airflow variables set model_type "RandomForest"
docker exec airflow-airflow-webserver-1 airflow variables set n_estimators "50"
docker exec airflow-airflow-webserver-1 airflow variables set max_depth "3"
docker exec airflow-airflow-webserver-1 airflow dags trigger titanic_ml_pipeline
```

**Experiment 2 — Random Forest (full depth)**
```powershell
docker exec airflow-airflow-webserver-1 airflow variables set model_type "RandomForest"
docker exec airflow-airflow-webserver-1 airflow variables set n_estimators "100"
docker exec airflow-airflow-webserver-1 airflow variables set max_depth "None"
docker exec airflow-airflow-webserver-1 airflow dags trigger titanic_ml_pipeline
```

**Experiment 3 — Logistic Regression**
```powershell
docker exec airflow-airflow-webserver-1 airflow variables set model_type "LogisticRegression"
docker exec airflow-airflow-webserver-1 airflow variables set C "0.5"
docker exec airflow-airflow-webserver-1 airflow variables set max_iter "300"
docker exec airflow-airflow-webserver-1 airflow dags trigger titanic_ml_pipeline
```

After all 3 runs go to **http://localhost:5000** → Titanic_Survival_Prediction → select all runs → Compare.

Graphs are automatically saved to `data/processed/graphs/` after each run. Comparison graphs update after every run from run 2 onwards.

---

## STEP 5 — Debugging

**View import errors:**
```powershell
docker exec airflow-airflow-webserver-1 airflow dags list-import-errors
```

**Test a single task:**
```powershell
docker exec airflow-airflow-worker-1 airflow tasks test titanic_ml_pipeline load_dataset 2024-01-01
docker exec airflow-airflow-worker-1 airflow tasks test titanic_ml_pipeline train_model 2024-01-01
```

**Check packages are installed:**
```powershell
docker exec airflow-airflow-worker-1 pip show mlflow
docker exec airflow-airflow-worker-1 pip show scikit-learn
```

**Check files are mounted:**
```powershell
docker exec airflow-airflow-worker-1 ls /opt/airflow/pipeline_scripts
docker exec airflow-airflow-worker-1 ls /opt/airflow/data
```

**Check MLflow is reachable from worker:**
```powershell
docker exec airflow-airflow-worker-1 curl http://mlflow:5000/health
```

---

## STEP 6 — Stop Everything

```powershell
docker compose down
```

Use `docker compose down -v` only if you want a completely clean slate — this deletes the database and your Airflow user account.

---

## Setting Variables via UI

Go to **http://localhost:8080** → Admin → Variables and add/edit keys manually.