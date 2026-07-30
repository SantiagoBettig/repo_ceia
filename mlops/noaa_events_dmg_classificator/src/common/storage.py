"""Utilidades de acceso a MinIO, compartidas por las distintas etapas del pipeline.

Colocar en: src/common/storage.py
Requiere también un src/common/__init__.py vacío para que sea un paquete importable.
"""

import os

import s3fs

# Estas variables ya están disponibles en los contenedores de Airflow (se las
# pasamos en el docker-compose.yml). Si corrés un script suelto fuera de
# Airflow, exportalas antes en tu terminal.
MINIO_ENDPOINT = os.environ["MLFLOW_S3_ENDPOINT_URL"]   # ej: http://minio:9000
MINIO_KEY = os.environ["AWS_ACCESS_KEY_ID"]
MINIO_SECRET = os.environ["AWS_SECRET_ACCESS_KEY"]


def obtener_fs():
    """Crea la conexión a MinIO vía s3fs, reutilizable en todo el pipeline."""
    return s3fs.S3FileSystem(
        key=MINIO_KEY,
        secret=MINIO_SECRET,
        client_kwargs={"endpoint_url": MINIO_ENDPOINT},
    )
