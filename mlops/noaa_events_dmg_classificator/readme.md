# MLOps Pipeline: NOAA Storm Events

Entorno de desarrollo de un ciclo de ML completo (extracción, features, entrenamiento y validación) para el trabajo de posgrado de la materia de MLOps (CEIA - FIUBA). Todos los componentes corren en contenedores Docker, orquestados con Docker Compose.

## Arquitectura

El entorno está compuesto por cuatro piezas principales:

- **MinIO:** data lake compatible con S3. Guarda tanto los datasets (crudos, procesados y splits) como los artifacts de MLflow.
- **PostgreSQL:** base de datos relacional, usada como backend de metadata tanto para Airflow (estado de DAGs) como para MLflow (parámetros, métricas y runs), en bases separadas dentro del mismo servidor.
- **MLflow:** tracking de experimentos y registro de modelos. Guarda metadata en PostgreSQL y artifacts (modelos, figuras) en MinIO.
- **Apache Airflow:** orquestador del pipeline. Ejecuta las tareas de cada etapa (extracción, features, entrenamiento, validación) como DAGs independientes, conectados mediante Airflow Datasets.

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
├── .env                      # credenciales locales (NO se commitea)
├── .env.example               # plantilla de variables, sin valores reales
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
│   ├── Dockerfile               # imagen custom: apache/airflow + libgomp1 + git + requirements.txt
│   ├── requirements.txt         # pandas, scikit-learn, scipy, joblib, lightgbm, optuna, mlflow, boto3, s3fs
│   └── dags/
│       ├── datasets_comunes.py  # definiciones compartidas de Airflow Datasets
│       ├── extraction_dag.py    # DAG de la etapa de extracción y limpieza
│       └── features_dag.py      # DAG de split temporal + feature engineering (tree_ohe)
│
└── src/
    ├── __init__.py
    ├── common/
    │   ├── __init__.py
    │   └── storage.py                        # obtener_fs(): conexión a MinIO, compartida por todas las etapas
    ├── extraction/
    │   ├── __init__.py
    │   └── extraction_storm_events.py         # consolidación, limpieza, features determinísticas y target
    ├── features/
    │   ├── __init__.py
    │   └── features_storm_events.py         # split temporal, roles de columnas y encoding (tree_ohe)
    ├── training/
    │   └── __init__.py                        # (pendiente)
    └── validation/
        └── __init__.py                        # (pendiente)
```

## Requisitos

- Docker Desktop con el backend **WSL2** habilitado (Settings → General → "Use the WSL 2 based engine")

## Puesta en marcha desde cero

1. Modifica `.env` para utilizar las credenciales que desees (contraseñas de MinIO/Postgres/Airflow, `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` iguales a las de MinIO, y un `AIRFLOW__WEBSERVER__SECRET_KEY` propio). Ten en cuenta que pueden usarse las que están por defecto
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

Este proyecto no incluye el dataset crudo en el repositorio (por tamaño). Antes de correr el pipeline por primera vez:

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
     -c "mc alias set local http://minio:9000 <tu_MINIO_ROOT_USER> <tu_MINIO_ROOT_PASSWORD> && mc cp --recursive /dataset/ local/datasets-crudos/"
   ```
5. Verificá la carga entrando a `http://localhost:9001` (usuario/contraseña de tu `.env`) → bucket `datasets-crudos`.

## Servicios

| Servicio | Puerto | Descripción |
|---|---|---|
| `minio` | 9000 (API), 9001 (consola web) | Servidor de object storage compatible con S3 |
| `postgres` | 5432 | Base de datos relacional (bases `airflow` y `mlflow` separadas) |
| `createbuckets` | — | Tarea de una sola vez: crea los buckets `mlflow-artifacts`, `datasets-generados`, `datasets-crudos` y `datasets-splits` |
| `mlflow` | 5000 | Servidor de tracking de experimentos y registro de modelos |
| `airflow-init` | — | Tarea de una sola vez: migra la base de Airflow y crea el usuario admin |
| `airflow-webserver` | 8080 | Interfaz web de Airflow |
| `airflow-scheduler` | — | Proceso que dispara y ejecuta las tareas de los DAGs (`LocalExecutor`) |

## Buckets en MinIO

| Bucket | Contenido |
|---|---|
| `datasets-crudos` | CSVs originales de NOAA (descargados de Kaggle) |
| `datasets-generados` | Datasets procesados por la etapa de extracción (`storm_events_clean_master.csv`, `storm_events_damage_modeling_base.csv`) |
| `datasets-splits` | Salida de la etapa de features: splits crudos (`storm_splits/`) y matrices `tree_ohe` + preprocessor + targets (`processed_datasets/`) |
| `mlflow-artifacts` | Artifacts de MLflow: modelos entrenados, figuras logueadas, etc. |

## DAGs de Airflow

El pipeline está separado en un DAG por etapa, conectados mediante **Airflow Datasets** (data-aware scheduling): cada DAG se dispara automáticamente cuando el DAG anterior actualiza el dataset del que depende, en vez de encadenarse por horario o de forma manual.

| DAG | Estado | Dispara cuando... | Produce |
|---|---|---|---|
| `extraction_dag` | ✅ Implementado y verificado | Manual (primer eslabón) | `DATASET_CLEAN` (`storm_events_damage_modeling_base.csv`) |
| `features_dag` | ✅ Implementado y verificado | `DATASET_CLEAN` se actualiza | `DATASET_FEATURES` (splits + `tree_ohe`) |
| `dag_3_entrenamiento` | ⏳ Pendiente | `DATASET_FEATURES` se actualiza | `DATASET_MODELO_ENTRENADO` |
| `dag_4_validacion` | ⏳ Pendiente | `DATASET_MODELO_ENTRENADO` se actualiza | — |

Las definiciones de los `Dataset` compartidos viven en `airflow/dags/datasets_comunes.py`, para asegurar que la URI coincida exactamente entre el DAG que produce y el que consume cada uno. Confirmado: al disparar `extraction_dag` manualmente, `features_dag` se dispara solo al terminar.


## Notas y observaciones

- **Contraseña de MinIO**: debe tener mínimo 8 caracteres.
- **`libgomp.so.1` faltante al importar LightGBM**: la imagen base de Airflow no trae esta librería del sistema; se instala con `apt-get install libgomp1` en el `Dockerfile` de `airflow/` (como `root`, antes de volver a `USER airflow`).
- **Logs de Airflow con `403 Forbidden`**: ocurre cuando el `webserver` y el `scheduler` generan cada uno una `secret_key` distinta al azar. Se resuelve fijando `AIRFLOW__WEBSERVER__SECRET_KEY` en el `.env` y pasándosela a los tres servicios de Airflow.
- **Warning de Git en MLflow** (`Failed to import Git...`): inofensivo, pero silenciarlo hace perder la trazabilidad del commit SHA en cada run. Se resolvió instalando `git` en el `Dockerfile` de `airflow/` en vez de silenciarlo con `GIT_PYTHON_REFRESH`.
- **Los links de MLflow en los logs de Airflow muestran `http://mlflow:5000/...` en vez de `localhost`**: es esperado — el script corre *dentro* del contenedor, donde `mlflow` es el hostname interno de Docker Compose. Para hacerlos clickeables desde el navegador de Windows sin editar el link a mano, se puede agregar `127.0.0.1 mlflow` a `C:\Windows\System32\drivers\etc\hosts`. Por ahora se dejó sin resolver (reemplazo manual al pegar el link).