"""
NOAA Storm Events — Entrenamiento del modelo LightGBM (baseline, balanceo,
búsqueda de hiperparámetros con Optuna y reentrenamiento final).

Convertido desde el notebook `3_storm_events_ml_models.ipynb`, tomando
ÚNICAMENTE las secciones correspondientes a LightGBM. Se descartaron los
demás modelos evaluados en el notebook original (regresión logística, SGD,
Decision Tree, Random Forest, XGBoost, CatBoost) — este proyecto entrena y
sirve exclusivamente LightGBM.

Cambios respecto al notebook original:
- Se removió el bloque de Google Colab (`drive.mount(...)`).
- Se removió el cacheo de modelos en disco (`if model_path.exists(): ...`):
  en un DAG de Airflow cada corrida entrena de cero, no tiene sentido
  "saltear" el entrenamiento basándose en archivos locales de un contenedor
  efímero.
- Solo se cargan los splits `train` y `validation` — el set de `test` queda
  reservado para la etapa de validación (dag_4), no se toca acá.
- Todo el entrenamiento (baseline, balanceo, Optuna, modelo final) se loguea
  a MLflow: el baseline como run de comparación, los trials de Optuna que
  mejoran el mejor resultado hasta el momento como runs anidados, y el
  modelo final registrado en el Model Registry de MLflow.
- No se cambió ningún criterio metodológico (hiperparámetros del baseline,
  estrategia de balanceo, espacio de búsqueda de Optuna, etc.).
"""

import json
import time

import matplotlib.pyplot as plt
import mlflow
import mlflow.lightgbm
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from scipy.sparse import load_npz
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.common.storage import obtener_fs


# ---------------------------------------------------------------------------
# 1. Configuración
# ---------------------------------------------------------------------------

BUCKET_SPLITS = "datasets-splits"
PROCESSED_PREFIX = "processed_datasets"
PROCESSED_PREFIX2 = "storm_splits"

MLFLOW_EXPERIMENT = "entrenamiento-lightgbm-storm-events"
MODEL_REGISTRY_NAME = "storm-events-lightgbm"

RANDOM_STATE = 42

# Estrategia de balanceo (undersample Clase 0, oversample Clases 3 y 4).
# Estos tamaños objetivo fueron definidos a partir de la distribución real
# del dataset NOAA Storm Events — si el dataset de entrada cambiara
# significativamente de tamaño, conviene revisar estos valores.
TARGET_SIZES = {
    0: 150_000,   # undersample
    1: 148_246,   # se mantiene (tamaño original)
    2: 126_534,   # se mantiene (tamaño original)
    3: 80_000,    # oversample
    4: 60_000,    # oversample
}

N_TRIALS = 30


# ---------------------------------------------------------------------------
# 2. Carga de datos (train / validation, desde el bucket)
# ---------------------------------------------------------------------------

def cargar_datos(fs, bucket=BUCKET_SPLITS, prefix=PROCESSED_PREFIX, prefix2=PROCESSED_PREFIX2):
    """Carga X_train/X_val (tree_ohe), y_train/y_val y los class weights."""
    def _load_npz(nombre):
        key = f"{bucket}/{prefix}/{nombre}.npz"
        with fs.open(key, "rb") as f:
            return load_npz(f)

    def _load_target(nombre):
        key = f"{bucket}/{prefix}/{nombre}.csv"
        with fs.open(key) as f:
            return pd.read_csv(f).squeeze()

    X_train = _load_npz("X_train_tree_ohe")
    X_val = _load_npz("X_val_tree_ohe")
    y_train = _load_target("y_train")
    y_val = _load_target("y_val")

    weights_key = f"{bucket}/{prefix2}/class_weights_train.json"
    with fs.open(weights_key) as f:
        cw_dict = {int(k): v for k, v in json.load(f).items()}

    print(f"X_train: {X_train.shape} | X_val: {X_val.shape}")
    print(f"Clases en y_train: {sorted(y_train.unique())}")
    print(f"Class weights: {cw_dict}")

    return X_train, X_val, y_train, y_val, cw_dict


# ---------------------------------------------------------------------------
# 3. Evaluación
# ---------------------------------------------------------------------------

def evaluate(name, y_true, y_pred, elapsed=None):
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    acc = accuracy_score(y_true, y_pred)

    time_str = f"{elapsed:.1f}s" if elapsed is not None else "N/A"

    print(f"\n{'=' * 55}")
    print(f"  {name}  |  Tiempo: {time_str}")
    print(f"{'=' * 55}")
    print(f"  Macro F1:    {macro_f1:.4f}")
    print(f"  Weighted F1: {weighted_f1:.4f}")
    print(f"  Accuracy:    {acc:.4f}")
    print(classification_report(y_true, y_pred, zero_division=0))

    return {
        "model": name,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "accuracy": acc,
        "time_s": elapsed,
    }


def graficar_matriz_confusion(y_true, y_pred, titulo):
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, colorbar=False, cmap="Blues", values_format=".2f")
    ax.set_title(titulo)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. Baseline (sin balancear, con class_weight)
# ---------------------------------------------------------------------------

def entrenar_baseline(X_train, y_train, cw_dict):
    print("Entrenando LightGBM baseline (sin balancear)...")
    t0 = time.time()

    modelo = LGBMClassifier(
        objective="multiclass",
        num_class=len(sorted(y_train.unique())),
        n_estimators=300,
        learning_rate=0.08,
        max_depth=-1,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight=cw_dict,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    modelo.fit(X_train, y_train)

    elapsed = time.time() - t0
    return modelo, elapsed


# ---------------------------------------------------------------------------
# 5. Balanceo de clases (undersample/oversample por índices)
# ---------------------------------------------------------------------------

def balancear_clases(X_train, y_train, target_sizes=TARGET_SIZES, random_state=RANDOM_STATE):
    """Undersample de la clase mayoritaria + oversample de las minoritarias,
    aplicado sobre los índices del train set (no sobre la matriz ya procesada)."""
    rng = np.random.RandomState(random_state)

    y_train_arr = y_train.to_numpy() if hasattr(y_train, "to_numpy") else np.array(y_train)
    indices_by_class = {c: np.where(y_train_arr == c)[0] for c in sorted(target_sizes)}

    for c, idx in indices_by_class.items():
        print(f"Clase {c}: {len(idx):,} muestras originales")

    balanced_indices = []
    for c, target_n in target_sizes.items():
        original_idx = indices_by_class[c]
        n_original = len(original_idx)

        if target_n <= n_original:
            sampled = rng.choice(original_idx, size=target_n, replace=False)
        else:
            sampled = rng.choice(original_idx, size=target_n, replace=True)

        balanced_indices.append(sampled)
        print(f"Clase {c}: {n_original:,} -> {target_n:,}")

    balanced_indices = np.concatenate(balanced_indices)
    rng.shuffle(balanced_indices)

    X_train_bal = X_train[balanced_indices]
    y_train_bal = y_train_arr[balanced_indices]

    print(f"\nTotal train balanceado: {len(balanced_indices):,}")
    print(f"Total train original:   {len(y_train_arr):,}")
    print("Verificación de balance:")
    unique, counts = np.unique(y_train_bal, return_counts=True)
    for c, cnt in zip(unique, counts):
        print(f"  Clase {c}: {cnt:,} ({cnt / len(y_train_bal) * 100:.1f}%)")

    return X_train_bal, y_train_bal


# ---------------------------------------------------------------------------
# 6. Búsqueda de hiperparámetros con Optuna
# ---------------------------------------------------------------------------

def build_objective(X_train_bal, y_train_bal, X_val, y_val, n_classes):
    def objective_lgbm(trial):
        params = {
            "objective": "multiclass",
            "num_class": n_classes,
            "num_leaves": trial.suggest_int("num_leaves", 15, 120),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.2, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 5.0, log=True),
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "verbose": -1,
        }

        modelo = LGBMClassifier(**params)
        modelo.fit(X_train_bal, y_train_bal)

        y_pred = modelo.predict(X_val)
        return f1_score(y_val, y_pred, average="macro", zero_division=0)

    return objective_lgbm


def build_mlflow_callback():
    """Callback de Optuna: loguea como run anidado de MLflow solo los trials
    que mejoraron el mejor resultado hasta ese momento."""
    def mlflow_callback(study, trial):
        if trial.value is None:
            return
        # study.best_trial ya refleja el mejor trial completado hasta ahora
        # (incluyendo el actual, porque el callback corre después del trial).
        if trial.number != study.best_trial.number:
            return  # este trial no mejoró el mejor resultado; no se loguea

        with mlflow.start_run(run_name=f"optuna-trial-{trial.number}-mejora", nested=True):
            mlflow.log_params(trial.params)
            mlflow.log_metric("macro_f1_val", trial.value)
            mlflow.set_tag("trial_number", trial.number)

    return mlflow_callback


def buscar_hiperparametros(X_train_bal, y_train_bal, X_val, y_val, n_classes, n_trials=N_TRIALS):
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=0)
    sampler = TPESampler(seed=RANDOM_STATE)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name="lightgbm_tuning",
    )

    objective = build_objective(X_train_bal, y_train_bal, X_val, y_val, n_classes)
    callback = build_mlflow_callback()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True, callbacks=[callback])

    print("Mejor trial LightGBM:")
    print(f"  Macro F1: {study.best_value:.4f}")
    print(f"  Params:   {study.best_params}")
    return study


# ---------------------------------------------------------------------------
# 7. Reentrenamiento final con los mejores hiperparámetros
# ---------------------------------------------------------------------------

def entrenar_final(X_train_bal, y_train_bal, best_params, n_classes):
    print("Entrenando LightGBM final (balanceado + hiperparámetros optimizados)...")
    t0 = time.time()

    modelo = LGBMClassifier(
        **best_params,
        objective="multiclass",
        num_class=n_classes,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    modelo.fit(X_train_bal, y_train_bal)

    elapsed = time.time() - t0
    return modelo, elapsed


# ---------------------------------------------------------------------------
# 8. main
# ---------------------------------------------------------------------------

def main():
    fs = obtener_fs()
    X_train, X_val, y_train, y_val, cw_dict = cargar_datos(fs)
    n_classes = len(sorted(y_train.unique()))

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    # --- 1. Baseline sin balancear ---
    with mlflow.start_run(run_name="baseline-sin-balancear"):
        modelo_baseline, elapsed = entrenar_baseline(X_train, y_train, cw_dict)
        y_pred_baseline = modelo_baseline.predict(X_val)
        resultado_baseline = evaluate("LightGBM (baseline)", y_val, y_pred_baseline, elapsed)

        mlflow.log_param("balanceado", False)
        mlflow.log_param("class_weight", "cw_dict (desde class_weights_train.json)")
        mlflow.log_metric("macro_f1", resultado_baseline["macro_f1"])
        mlflow.log_metric("weighted_f1", resultado_baseline["weighted_f1"])
        mlflow.log_metric("accuracy", resultado_baseline["accuracy"])
        mlflow.log_metric("tiempo_entrenamiento_s", elapsed)

        fig_baseline = graficar_matriz_confusion(
            y_val, y_pred_baseline, "Matriz de confusión LightGBM baseline"
        )
        mlflow.log_figure(fig_baseline, "matriz_confusion_baseline.png")
        plt.close(fig_baseline)

        # Registramos el baseline como el modelo 
        mlflow.lightgbm.log_model(
            modelo_baseline,
            artifact_path="model",
            registered_model_name=MODEL_REGISTRY_NAME,
        )

    print("\nEntrenamiento completo.")
    print(f"Baseline  Macro F1 (val): {resultado_baseline['macro_f1']:.4f}")


if __name__ == "__main__":
    main()