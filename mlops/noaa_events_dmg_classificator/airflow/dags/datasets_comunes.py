"""Definiciones compartidas de Airflow Datasets para el pipeline de Storm Events.

Importante: la URI debe ser idéntica en el DAG que lo produce (outlets=[...])
y en el que lo consume (schedule=[...]). Airflow los matchea por string exacta,
no verifica que el archivo realmente exista en esa ruta.
"""

from airflow.datasets import Dataset

DATASET_CLEAN = Dataset("s3://datasets-generados/storm_events_clean_master.csv")
DATASET_FEATURES = Dataset("s3://datasets-generados/storm_events_features.csv")
DATASET_MODELO_ENTRENADO = Dataset("mlflow://storm-events-damage/modelo-entrenado")