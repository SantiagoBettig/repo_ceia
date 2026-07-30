from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

from src.features.features_storm_events import main as construir_features
from datasets_comunes import DATASET_CLEAN, DATASET_FEATURES

with DAG(
    dag_id="features_dag",
    description="Split temporal, roles de columnas y feature engineering (tree_ohe) para LightGBM",
    schedule=[DATASET_CLEAN],   # se dispara solo cuando extraction_dag actualiza este dataset
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["ceia", "storm-events", "features"],
) as dag:

    tarea_features = PythonOperator(
        task_id="split_y_feature_engineering",
        python_callable=construir_features,
        outlets=[DATASET_FEATURES],
    )