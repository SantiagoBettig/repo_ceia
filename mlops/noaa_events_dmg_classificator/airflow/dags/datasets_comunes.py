"""Definiciones compartidas de Airflow Datasets para el pipeline de Storm Events.

Importante: la URI debe ser idéntica en el DAG que lo produce (outlets=[...])
y en el que lo consume (schedule=[...]) — Airflow los matchea por string exacta,
no verifica que el archivo realmente exista en esa ruta.

Colocar en: airflow/dags/datasets_comunes.py
"""

from airflow.datasets import Dataset

# Producido por dag_1_extraccion. Es el archivo que dag_2_features realmente lee.
DATASET_CLEAN = Dataset("s3://datasets-generados/storm_events_damage_modeling_base.csv")

# Producido por dag_2_features. URI simbólica: representa el conjunto de salidas
# de esta etapa (X_train/val/test_tree_ohe.npz, preprocessor, y_train/val/test.csv),
# no un único archivo — alcanza con que sea consistente entre outlet y schedule.
DATASET_FEATURES = Dataset("s3://datasets-splits/processed_datasets/")

# Producido por dag_3_entrenamiento. URI lógica (no es una ruta real de archivo),
# ya que el artifact vive registrado en MLflow, no en una ruta de S3 tradicional.
DATASET_MODELO_ENTRENADO = Dataset("mlflow://storm-events-damage/modelo-entrenado")