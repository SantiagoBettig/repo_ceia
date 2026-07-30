"""
NOAA Storm Events — Split temporal, roles de columnas y preparación segura para modelado.

Convertido desde el notebook `2_storm_events_split_final.ipynb` a un script con
funciones, siguiendo el mismo patrón que `extraccion_storm_events.py`.

Cambios respecto al notebook original:
- Se removió el bloque de Google Colab (`drive.mount(...)`).
- Se removieron las variantes `scaled_ohe` (modelos lineales/SVM/KNN) y `catboost`
  (categóricas nativas) — según lo indicado, solo se necesita la variante
  `tree_ohe` para LightGBM. El preprocessor quedó simplificado: ya no tiene la
  rama de escalado (`scale_numeric`), porque nunca se usa.
- Lectura y escritura van contra buckets de MinIO (vía `s3fs`), no contra disco
  local. La conexión (`obtener_fs`) vive en `src/common/storage.py`.
- No se cambió ningún criterio metodológico (roles de columnas, ventanas
  temporales del split, cálculo de class weights, etc.).
"""

import json

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
from scipy.sparse import save_npz
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.common.storage import obtener_fs


# ---------------------------------------------------------------------------
# 1. Configuración
# ---------------------------------------------------------------------------

BUCKET_ENTRADA = "datasets-generados"
INPUT_KEY = f"{BUCKET_ENTRADA}/storm_events_damage_modeling_base.csv"

BUCKET_SALIDA = "datasets-splits"
SPLITS_PREFIX = "storm_splits"
PROCESSED_PREFIX = "processed_datasets"

# NOAA amplió fuertemente la cobertura de eventos a partir de 1996.
MIN_MODEL_YEAR = 1996
TRAIN_END_YEAR = 2015
VAL_START_YEAR = 2016
VAL_END_YEAR = 2017
TEST_START_YEAR = 2018
TEST_END_YEAR = 2019
# Los años posteriores se conservan como información reciente, pero no se
# usan para evaluar el baseline porque pueden estar incompletos.
RECENT_HOLDOUT_START_YEAR = 2020

RANDOM_STATE = 42

TARGET = "DAMAGE_CLASS"

ID_COLUMNS = ["EVENT_ID", "EPISODE_ID"]

# Nunca deben entrar al modelo.
FORBIDDEN_FEATURES = [
    "DAMAGE_REAL_2020",
    "TOTAL_DEATHS",
    "TOTAL_INJURIES",
    "TOTAL_CASUALTIES",
    "HAS_FATALITIES",
    "HAS_CASUALTIES",
]

# Variables categóricas nominales. Se encodean después del split.
CATEGORICAL_FEATURES = [
    "EVENT_TYPE",
    "CZ_TYPE",
    "STATE",
    "REGION",
    "WFO",
    "SOURCE",
    "SEASON",
    "TIME_OF_DAY",
    "MAGNITUDE_TYPE",
    "FLOOD_CAUSE",
]

# Variables numéricas generales. Su imputador se ajusta solo con train.
GENERAL_NUMERIC_FEATURES = [
    "YEAR",
    "MONTH",
    "HOUR",
    "DAY_OF_WEEK",
    "DAY_OF_YEAR",
    "DURATION_MIN",
    "BEGIN_LAT",
    "BEGIN_LON",
    "END_LAT",
    "END_LON",
    "LAT_BIN",
    "LON_BIN",
]

# Variables cuyo NaN suele significar "no aplica" o "no informado".
# Se imputan con 0, acompañadas por flags de presencia (si aplica).
STRUCTURAL_ZERO_FEATURES = [
    "TRACK_DISTANCE_KM",
    "MAGNITUDE",
    "TOR_SCALE_NUM",
    "TOR_LENGTH",
    "TOR_WIDTH",
    "TOR_AREA_KM2",
]

# Flags ya generados en la etapa de extracción.
BINARY_FEATURES = [
    "HAS_COORDINATES",
    "HAS_TRACK_DISTANCE",
    "HAS_MAGNITUDE",
    "HAS_TORNADO_DATA",
]


# ---------------------------------------------------------------------------
# 2. Carga y chequeos iniciales
# ---------------------------------------------------------------------------

def cargar_dataset_modelado(fs, input_key=INPUT_KEY):
    """Lee la base de modelado (generada por la etapa de extracción) desde el bucket."""
    with fs.open(input_key) as f:
        df = pd.read_csv(f, low_memory=False)

    print(f"Filas:    {df.shape[0]:,}")
    print(f"Columnas: {df.shape[1]}")
    print("\nColumnas disponibles:")
    for idx, column in enumerate(df.columns, start=1):
        print(f"  {idx:2d}. {column} ({df[column].dtype})")
    return df


def chequeos_iniciales(df):
    """Valida que el dataset de entrada tenga la forma esperada antes de continuar."""
    required_columns = {
        "EVENT_ID", "EPISODE_ID", "YEAR", "EVENT_TYPE", "CZ_TYPE",
        "DAMAGE_REAL_2020", "DAMAGE_CLASS",
    }
    missing_required = sorted(required_columns - set(df.columns))
    assert not missing_required, f"Faltan columnas requeridas: {missing_required}"

    assert df["EVENT_ID"].is_unique, "EVENT_ID debe ser único."
    assert df["DAMAGE_CLASS"].notna().all(), "La base de modelado debe tener target completo."
    assert df["DAMAGE_REAL_2020"].notna().all(), "La base de modelado debe tener daño reportado."

    print("Chequeos iniciales superados.")


# ---------------------------------------------------------------------------
# 3. Roles de columnas
# ---------------------------------------------------------------------------

def existing(columns, df):
    """Filtra una lista de nombres de columna, conservando solo las presentes en df."""
    return [column for column in columns if column in df.columns]


def resolver_roles_columnas(df):
    """Filtra las listas de roles contra las columnas realmente presentes en df,
    y arma BASELINE_FEATURES a partir de ellas."""
    id_columns = existing(ID_COLUMNS, df)
    forbidden_features = existing(FORBIDDEN_FEATURES, df)
    categorical_features = existing(CATEGORICAL_FEATURES, df)
    general_numeric_features = existing(GENERAL_NUMERIC_FEATURES, df)
    structural_zero_features = existing(STRUCTURAL_ZERO_FEATURES, df)
    binary_features = existing(BINARY_FEATURES, df)

    baseline_features = (
        categorical_features
        + general_numeric_features
        + structural_zero_features
        + binary_features
    )

    print(f"Features categóricas:            {len(categorical_features)}")
    print(f"Features numéricas generales:    {len(general_numeric_features)}")
    print(f"Features con imputación a cero:  {len(structural_zero_features)}")
    print(f"Flags binarios:                  {len(binary_features)}")
    print(f"Total baseline:                  {len(baseline_features)}")

    return {
        "id_columns": id_columns,
        "forbidden_features": forbidden_features,
        "categorical_features": categorical_features,
        "general_numeric_features": general_numeric_features,
        "structural_zero_features": structural_zero_features,
        "binary_features": binary_features,
        "baseline_features": baseline_features,
    }


# ---------------------------------------------------------------------------
# 4. Auditoría de nulos estructurales (no transforma nada, solo informa)
# ---------------------------------------------------------------------------

def summarize_structural_feature(df, value_col, flag_col=None):
    print("=" * 70)
    print(value_col)
    print("=" * 70)
    print(f"Nulos:             {df[value_col].isna().sum():,} ({df[value_col].isna().mean() * 100:.2f}%)")
    print(f"Ceros informados:  {df[value_col].eq(0).sum():,} ({df[value_col].eq(0).mean() * 100:.2f}%)")
    if flag_col and flag_col in df.columns:
        print(f"{flag_col}=1: {df[flag_col].eq(1).sum():,} ({df[flag_col].eq(1).mean() * 100:.2f}%)")
        print(f"Consistencia flag vs notna: {(df[flag_col].eq(1) == df[value_col].notna()).mean() * 100:.2f}%")
    print()


def auditoria_nulos_estructurales(df):
    """Verifica por qué ciertas variables tienen muchos faltantes (no transforma nada)."""
    summarize_structural_feature(df, "TRACK_DISTANCE_KM", "HAS_TRACK_DISTANCE")
    summarize_structural_feature(df, "MAGNITUDE", "HAS_MAGNITUDE")
    summarize_structural_feature(df, "TOR_AREA_KM2", "HAS_TORNADO_DATA")

    for col in ["FLOOD_CAUSE", "MAGNITUDE_TYPE", "CZ_TYPE"]:
        if col in df.columns:
            print("=" * 70)
            print(col)
            print("=" * 70)
            print(f"Nulos: {df[col].isna().sum():,} ({df[col].isna().mean() * 100:.2f}%)")
            print(df[col].value_counts(dropna=False).head(15))
            print()


# ---------------------------------------------------------------------------
# 5-6. Ventanas temporales y split
# ---------------------------------------------------------------------------

def definir_ventanas_temporales(df):
    """Separa histórico (< MIN_MODEL_YEAR), ventana principal, y holdout reciente."""
    historical_df = df[df["YEAR"] < MIN_MODEL_YEAR].copy()
    modeling_window_df = df[df["YEAR"].between(MIN_MODEL_YEAR, TEST_END_YEAR)].copy()
    recent_holdout_df = df[df["YEAR"] >= RECENT_HOLDOUT_START_YEAR].copy()

    print(f"Histórico anterior a {MIN_MODEL_YEAR}: {len(historical_df):,} filas")
    print(f"Ventana principal {MIN_MODEL_YEAR}-{TEST_END_YEAR}: {len(modeling_window_df):,} filas")
    print(f"Holdout reciente desde {RECENT_HOLDOUT_START_YEAR}: {len(recent_holdout_df):,} filas")
    return historical_df, modeling_window_df, recent_holdout_df


def split_temporal(modeling_window_df, recent_holdout_df):
    """Split temporal reproducible: train / validation / test / recent_holdout."""
    train_df = modeling_window_df[modeling_window_df["YEAR"] <= TRAIN_END_YEAR].copy()
    val_df = modeling_window_df[modeling_window_df["YEAR"].between(VAL_START_YEAR, VAL_END_YEAR)].copy()
    test_df = modeling_window_df[modeling_window_df["YEAR"].between(TEST_START_YEAR, TEST_END_YEAR)].copy()

    split_frames = {
        "train": train_df,
        "validation": val_df,
        "test": test_df,
        "recent_holdout": recent_holdout_df,
    }

    for split_name, split_df in split_frames.items():
        anio_min = split_df["YEAR"].min() if len(split_df) else "-"
        anio_max = split_df["YEAR"].max() if len(split_df) else "-"
        print(f"{split_name:<14}: {len(split_df):>10,} filas | años {anio_min}-{anio_max}")

    return train_df, val_df, test_df, split_frames


def target_distribution(df, name):
    counts = df[TARGET].value_counts().sort_index()
    pct = (counts / len(df) * 100).round(2)
    result = pd.DataFrame({"count": counts, "pct": pct})
    result.index.name = TARGET
    print(f"\n{name.upper()} — {len(df):,} filas")
    print(result)
    return result


def graficar_distribucion_target(train_df, val_df, test_df):
    """Grafica la distribución porcentual del target por split, para loguear a MLflow."""
    dist = pd.DataFrame({
        "train": train_df[TARGET].value_counts(normalize=True).sort_index() * 100,
        "validation": val_df[TARGET].value_counts(normalize=True).sort_index() * 100,
        "test": test_df[TARGET].value_counts(normalize=True).sort_index() * 100,
    })
    fig, ax = plt.subplots(figsize=(9, 5))
    dist.plot.bar(ax=ax)
    ax.set_title("Distribución de DAMAGE_CLASS por split")
    ax.set_xlabel("Clase")
    ax.set_ylabel("% dentro del split")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 7. Exportación de splits crudos (sin imputar, sin OHE, sin escalar)
# ---------------------------------------------------------------------------

def exportar_splits_crudos(fs, split_frames, roles, bucket=BUCKET_SALIDA, prefix=SPLITS_PREFIX):
    """Exporta cada split tal cual (sin transformar), preservando metadata y target."""
    export_columns = list(dict.fromkeys(
        roles["id_columns"] + roles["baseline_features"] + roles["forbidden_features"] + [TARGET]
    ))

    for split_name, split_df in split_frames.items():
        columnas_presentes = [c for c in export_columns if c in split_df.columns]
        key = f"{bucket}/{prefix}/storm_{split_name}_raw.csv"
        with fs.open(key, "w") as f:
            split_df[columnas_presentes].to_csv(f, index=False)
        print(f"Exportado: s3://{key} ({len(split_df):,} filas × {len(columnas_presentes)} columnas)")


# ---------------------------------------------------------------------------
# 8. Class weights (calculados únicamente con train)
# ---------------------------------------------------------------------------

def calcular_class_weights(train_df):
    train_counts = train_df[TARGET].value_counts().sort_index()
    n_train = len(train_df)
    n_classes = train_counts.size
    class_weights = (n_train / (n_classes * train_counts)).to_dict()
    class_weights = {int(cls): float(weight) for cls, weight in class_weights.items()}

    print("Class weights basados únicamente en train:")
    for cls, weight in class_weights.items():
        print(f"  Clase {cls}: {weight:.4f}")
    return class_weights


def graficar_class_weights(class_weights):
    fig, ax = plt.subplots(figsize=(6, 4))
    clases = list(class_weights.keys())
    pesos = list(class_weights.values())
    ax.bar([str(c) for c in clases], pesos, color="slateblue")
    ax.set_title("Class weights (calculados solo con train)")
    ax.set_xlabel("Clase")
    ax.set_ylabel("Peso")
    plt.tight_layout()
    return fig


def exportar_class_weights(fs, class_weights, bucket=BUCKET_SALIDA, prefix=SPLITS_PREFIX):
    key = f"{bucket}/{prefix}/class_weights_train.json"
    with fs.open(key, "w") as f:
        json.dump(class_weights, f, indent=2)
    print(f"Exportado: s3://{key}")


# ---------------------------------------------------------------------------
# 9. Preprocessor para modelos de árbol (OHE, sin escalado)
# ---------------------------------------------------------------------------

def build_ohe_preprocessor(roles):
    """Preprocessor para modelos basados en árboles (LightGBM): OHE sin escalado.

    - Categóricas: imputación con "NotApplicable_or_Missing" + OHE.
    - Numéricas generales: imputación con la mediana de train.
    - Numéricas estructurales: imputación con 0.
    - Flags binarios: se conservan sin cambios (passthrough).
    """
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="NotApplicable_or_Missing")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", min_frequency=100)),
    ])

    general_numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    structural_numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
    ])

    return ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, roles["categorical_features"]),
            ("numeric_general", general_numeric_pipeline, roles["general_numeric_features"]),
            ("numeric_structural", structural_numeric_pipeline, roles["structural_zero_features"]),
            ("binary", "passthrough", roles["binary_features"]),
        ],
        remainder="drop",
    )


def generar_dataset_tree_ohe(fs, preprocessor, X_train, X_val, X_test, y_train, y_val, y_test,
                              bucket=BUCKET_SALIDA, prefix=PROCESSED_PREFIX):
    """Ajusta el preprocessor SOLO con train, transforma los tres splits, y exporta
    matrices dispersas + preprocessor + targets al bucket."""
    print("Generando versión OHE sin escalado (tree_ohe)...")

    X_train_tree = preprocessor.fit_transform(X_train)
    X_val_tree = preprocessor.transform(X_val)
    X_test_tree = preprocessor.transform(X_test)

    for nombre, matriz in [("X_train_tree_ohe", X_train_tree),
                            ("X_val_tree_ohe", X_val_tree),
                            ("X_test_tree_ohe", X_test_tree)]:
        key = f"{bucket}/{prefix}/{nombre}.npz"
        with fs.open(key, "wb") as f:
            save_npz(f, matriz)
        print(f"Exportado: s3://{key}")

    preprocessor_key = f"{bucket}/{prefix}/preprocessor_tree_ohe.joblib"
    with fs.open(preprocessor_key, "wb") as f:
        joblib.dump(preprocessor, f)
    print(f"Exportado: s3://{preprocessor_key}")

    for nombre, serie in [("y_train", y_train), ("y_val", y_val), ("y_test", y_test)]:
        key = f"{bucket}/{prefix}/{nombre}.csv"
        with fs.open(key, "w") as f:
            serie.to_csv(f, index=False)
        print(f"Exportado: s3://{key}")

    print("Train:", X_train_tree.shape)
    print("Validation:", X_val_tree.shape)
    print("Test:", X_test_tree.shape)

    return X_train_tree, X_val_tree, X_test_tree


# ---------------------------------------------------------------------------
# 10. main
# ---------------------------------------------------------------------------

def main():
    fs = obtener_fs()

    df = cargar_dataset_modelado(fs)
    chequeos_iniciales(df)
    roles = resolver_roles_columnas(df)
    auditoria_nulos_estructurales(df)

    _historical_df, modeling_window_df, recent_holdout_df = definir_ventanas_temporales(df)
    train_df, val_df, test_df, split_frames = split_temporal(modeling_window_df, recent_holdout_df)

    target_distribution(train_df, "train")
    target_distribution(val_df, "validation")
    target_distribution(test_df, "test")

    X_train = train_df[roles["baseline_features"]].copy()
    X_val = val_df[roles["baseline_features"]].copy()
    X_test = test_df[roles["baseline_features"]].copy()
    y_train = train_df[TARGET].copy()
    y_val = val_df[TARGET].copy()
    y_test = test_df[TARGET].copy()

    exportar_splits_crudos(fs, split_frames, roles)

    class_weights = calcular_class_weights(train_df)
    exportar_class_weights(fs, class_weights)

    preprocessor = build_ohe_preprocessor(roles)
    generar_dataset_tree_ohe(fs, preprocessor, X_train, X_val, X_test, y_train, y_val, y_test)

    # --- Logging a MLflow: resumen de esta etapa ---
    mlflow.set_experiment("feature-engineering-storm-events")
    with mlflow.start_run(run_name="split-y-tree-ohe"):
        mlflow.log_param("min_model_year", MIN_MODEL_YEAR)
        mlflow.log_param("train_end_year", TRAIN_END_YEAR)
        mlflow.log_param("val_years", f"{VAL_START_YEAR}-{VAL_END_YEAR}")
        mlflow.log_param("test_years", f"{TEST_START_YEAR}-{TEST_END_YEAR}")
        mlflow.log_metric("n_train", len(train_df))
        mlflow.log_metric("n_val", len(val_df))
        mlflow.log_metric("n_test", len(test_df))

        fig_dist = graficar_distribucion_target(train_df, val_df, test_df)
        mlflow.log_figure(fig_dist, "distribucion_target_por_split.png")
        plt.close(fig_dist)

        fig_weights = graficar_class_weights(class_weights)
        mlflow.log_figure(fig_weights, "class_weights_train.png")
        plt.close(fig_weights)


if __name__ == "__main__":
    main()
