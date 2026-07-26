# MLOps Pipeline: NOAA Storm Events

Entorno de desarrollo de un ciclo de ML completo (creación de dataset maestro, extracción y creación de features, entrenamiento de modelo y validación) para el trabajo de posgrado de la materia de MLOps (CEIA - FIUBA). Todos los componentes corren en contenedores Docker, orquestados con Docker Compose.

## Arquitectura

El entorno está compuesto por cuatro piezas principales:

- **MinIO:** data lake compatible con S3. Se guardan tanto los datasets (crudos y procesados) como los artifacts de MLflow.
- **PostgreSQL:** base de datos relacional, usada como backend de metadata tanto para Airflow (estado de DAGs) como para MLflow (parámetros, métricas y runs), en bases separadas dentro del mismo servidor.
- **MLflow:** tracking de experimentos y registro de modelos. Guarda metadata en PostgreSQL y artifacts (modelos, figuras) en MinIO.
- **Apache Airflow:** orquestador del pipeline. Ejecuta las tareas de cada etapa (extracción, features, entrenamiento, validación) como DAGs independientes.

```
Airflow  →  MLflow  →  PostgreSQL (metadata)
                    →  MinIO (artifacts)
Airflow  →  PostgreSQL (metadata propia)
Airflow  →  MinIO (lectura/escritura de datasets)
```

## Estructura de carpetas

```
mlops-pipeline/
├── docker-compose.yml
├── .env                      # credenciales locales
├── .gitignore
├── README.md
│
├── dataset/                    # CSVs crudos descargados de Kaggle (NO se commitea)
│
├── postgres/
│   └── init-multiple-dbs.sh    # crea las bases "airflow" y "mlflow" al iniciar Postgres
│
├── mlflow/
│   └── Dockerfile              # imagen custom: mlflow + psycopg2-binary + boto3
│
├── airflow/
│   ├── Dockerfile               # imagen custom: apache/airflow + libgomp1 + requirements.txt
│   ├── requirements.txt         # pandas, scikit-learn, lightgbm, optuna, mlflow, boto3, s3fs
│   └── dags/
│       ├── datasets_comunes.py  # definiciones compartidas de Airflow Datasets
│       └── extraction_dag.py  # DAG de la etapa de extracción y limpieza
│
└── src/
    ├── __init__.py
    ├── extraction/
    │   ├── __init__.py
    │   └── extraction_storm_events.py   # consolidación, limpieza, features determinísticas y target
    ├── features/
    │   └── __init__.py                  # (pendiente)
    ├── training/
    │   └── __init__.py                  # (pendiente)
    └── validation/
        └── __init__.py                  # (pendiente)
```

## Requisitos

- Docker Desktop con el backend **WSL2** habilitado (Settings → General → "Use the WSL 2 based engine")

## Puesta en marcha desde cero

1. Completá en .env tus propias credenciales (contraseñas de MinIO/Postgres/Airflow, `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` iguales a las de MinIO, y un `AIRFLOW__WEBSERVER__SECRET_KEY` propio). Si se desea, pueden usarse las credenciales por defecto.
2. Levantá el entorno:
   ```powershell
   docker compose up -d --build
   ```
3. Verificá que todos los servicios estén sanos:
   ```powershell
   docker compose ps -a
   ```
   - `minio`, `postgres` → `Up (healthy)`
   - `createbuckets`, `airflow-init` → `Exited (0)`
   - `mlflow`, `airflow-webserver`, `airflow-scheduler` → `Up`

### Carga del dataset crudo (una sola vez)

Este proyecto no incluye el dataset crudo en el repositorio (por tamaño). Antes de correr el pipeline por primera vez deben cargarse los datasets dentro del bucket "datasets-crudos":

1. Descargá el dataset desde Kaggle: https://www.kaggle.com/datasets/crawford/noaa-storm-events-database
2. Descomprimilo dentro de la carpeta `dataset/` en la raíz de este repo (creála si no existe). No hace falta preservar ninguna estructura de subcarpetas particular, el pipeline busca los CSVs por nombre en todo el bucket.
3. Confirmá el nombre de la red que generó Docker Compose en tu máquina:
   ```powershell
   docker network ls
   ```
   Buscá una red con el patrón `<nombre_de_tu_carpeta>_default`.
4. Parado en la raíz del repo, subí los CSVs al bucket:
   ```powershell
   docker run --rm -it `
     --network <NOMBRE_DE_RED_DEL_PASO_3> `
     -v ${PWD}/dataset:/dataset `
     --entrypoint sh `
     minio/mc:latest `
     -c "mc alias set local http://minio:9000 admin <tu_MINIO_ROOT_PASSWORD> && mc cp --recursive /dataset/ local/datasets-crudos/"
   ```
5. Verificá la carga entrando a `http://localhost:9001` (usuario/contraseña de tu `.env`) → bucket `datasets-crudos`.

## Servicios

| Servicio | Puerto | Descripción |
|---|---|---|
| `minio` | 9000 (API), 9001 (consola web) | Servidor de object storage compatible con S3 |
| `postgres` | 5432 | Base de datos relacional (bases `airflow` y `mlflow` separadas) |
| `createbuckets` | - | Tarea de una sola vez: crea los buckets `mlflow-artifacts`, `datasets-generados` y `datasets-crudos` |
| `mlflow` | 5000 | Servidor de tracking de experimentos y registro de modelos |
| `airflow-init` | - | Tarea de una sola vez: migra la base de Airflow y crea el usuario admin |
| `airflow-webserver` | 8080 | Interfaz web de Airflow |
| `airflow-scheduler` | - | Proceso que dispara y ejecuta las tareas de los DAGs (`LocalExecutor`) |

## Buckets en MinIO

| Bucket | Contenido |
|---|---|
| `datasets-crudos` | CSVs originales de NOAA (descargados de Kaggle) |
| `datasets-generados` | Datasets procesados por el pipeline (`storm_events_clean_master.csv`, `storm_events_damage_modeling_base.csv`, y los que se agreguen en etapas futuras) |
| `mlflow-artifacts` | Artifacts de MLflow: modelos entrenados, figuras de EDA, etc. |

## DAGs de Airflow

El pipeline está separado en un DAG por etapa, conectados mediante **Airflow Datasets** (data-aware scheduling): cada DAG se dispara automáticamente cuando el DAG anterior actualiza el dataset del que depende, en vez de encadenarse por horario o de forma manual.

| DAG | Estado | Dispara cuando... | Produce |
|---|---|---|---|
| `extraction_dag` | ✅ Implementado y verificado | Manual (primer eslabón) | `DATASET_CLEAN` (`storm_events_clean_master.csv`) |
| `dag_2_features` | ⏳ Pendiente | `DATASET_CLEAN` se actualiza | `DATASET_FEATURES` |
| `dag_3_entrenamiento` | ⏳ Pendiente | `DATASET_FEATURES` se actualiza | `DATASET_MODELO_ENTRENADO` |
| `dag_4_validacion` | ⏳ Pendiente | `DATASET_MODELO_ENTRENADO` se actualiza | — |

Las definiciones de los `Dataset` compartidos viven en `airflow/dags/datasets_comunes.py`, para asegurar que la URI coincida exactamente entre el DAG que produce y el que consume cada uno.
