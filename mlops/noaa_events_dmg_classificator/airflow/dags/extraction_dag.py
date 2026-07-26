from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

from src.extraction.extraction_storm_events import main as extraer_datos
from datasets_comunes import DATASET_CLEAN


with DAG(
    dag_id="extraction_dag",
    description="Consolida, limpia y arma el dataset maestro de NOAA Storm Events",
    schedule=None,          # disparo manual; es el primer eslabón de la cadena
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["ceia", "storm-events", "extraccion"],
) as dag:

    tarea_extraccion = PythonOperator(
        task_id="extraccion_y_limpieza",
        python_callable=extraer_datos,
        outlets=[DATASET_CLEAN],
    )