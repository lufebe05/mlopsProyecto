"""
Módulo de entrenamiento, evaluación y registro de modelos con MLflow.
Entrena modelos base y no lineales comparando métricas en escala original de rupias.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Tuple
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Silenciar logs verbosos de MLflow y advertencias para mantener la terminal limpia
warnings.filterwarnings("ignore")
os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "false"
logging.getLogger("mlflow").setLevel(logging.ERROR)

import mlflow
import mlflow.sklearn

# Rutas reproducibles
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "mlruns_artifacts"
SQLITE_DB_PATH = PROJECT_ROOT / "mlflow.db"

EXPERIMENT_NAME = "Prediccion_Precios_Vehiculos"


def load_processed_data() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Carga los conjuntos procesados de entrenamiento y prueba."""
    train_path = PROCESSED_DATA_DIR / "train_processed.csv"
    test_path = PROCESSED_DATA_DIR / "test_processed.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("[Error] Archivos procesados no encontrados en data/processed/.")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=["selling_price_log"])
    y_train_log = train_df["selling_price_log"]

    X_test = test_df.drop(columns=["selling_price_log"])
    y_test_log = test_df["selling_price_log"]

    print(f"[Info] Datos cargados: {X_train.shape[0]} train, {X_test.shape[0]} test ({X_train.shape[1]} variables)")
    return X_train, y_train_log, X_test, y_test_log


def evaluate_predictions(y_true_log: pd.Series, y_pred_log: np.ndarray) -> Dict[str, float]:
    """Calcula métricas de desempeño sobre escala logarítmica y escala original en INR."""
    y_true_real = np.expm1(y_true_log)
    y_pred_real = np.expm1(y_pred_log)

    rmse_real = float(np.sqrt(mean_squared_error(y_true_real, y_pred_real)))
    mae_real = float(mean_absolute_error(y_true_real, y_pred_real))
    r2_real = float(r2_score(y_true_real, y_pred_real))

    rmse_log = float(np.sqrt(mean_squared_error(y_true_log, y_pred_log)))
    r2_log = float(r2_score(y_true_log, y_pred_log))

    return {
        "rmse_inr": rmse_real,
        "mae_inr": mae_real,
        "r2_score_real": r2_real,
        "rmse_log": rmse_log,
        "r2_score_log": r2_log,
    }


def train_and_log_model(
    model_name: str,
    model_instance: Any,
    params: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train_log: pd.Series,
    X_test: pd.DataFrame,
    y_test_log: pd.Series,
):
    """Entrena un estimador y registra parámetros, métricas y artefacto en MLflow."""
    with mlflow.start_run(run_name=model_name):
        print(f"\n--- Iniciando Entrenamiento: {model_name} ---")

        # Ajuste del modelo
        model_instance.fit(X_train, y_train_log)

        # Predicciones
        y_train_pred = model_instance.predict(X_train)
        y_test_pred = model_instance.predict(X_test)

        # Cálculo de métricas
        metrics_train = evaluate_predictions(y_train_log, y_train_pred)
        metrics_test = evaluate_predictions(y_test_log, y_test_pred)

        # Registro en MLflow
        mlflow.log_params(params)

        for k, v in metrics_test.items():
            mlflow.log_metric(f"test_{k}", v)
        for k, v in metrics_train.items():
            mlflow.log_metric(f"train_{k}", v)

        # Registro serializado usando pickle tradicional (evita el bloqueo de skops)
        mlflow.sklearn.log_model(
            sk_model=model_instance,
            name="model",
            serialization_format="pickle",
        )

        model_path = MODELS_DIR / f"{model_name.lower().replace(' ', '_')}.joblib"
        joblib.dump(model_instance, model_path)
        mlflow.log_artifact(str(model_path), artifact_path="local_checkpoints")

        print(f"[Resultados {model_name}]")
        print(f"  Test RMSE (INR): {metrics_test['rmse_inr']:,.2f}")
        print(f"  Test MAE (INR):  {metrics_test['mae_inr']:,.2f}")
        print(f"  Test R²:         {metrics_test['r2_score_real']:.4f}")
        print(f"  Test R² (Log):   {metrics_test['r2_score_log']:.4f}")


def run_experiment_pipeline():
    """Configura el backend SQLite de MLflow y corre la experimentación."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    tracking_uri = f"sqlite:///{SQLITE_DB_PATH.as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        client.create_experiment(
            name=EXPERIMENT_NAME,
            artifact_location=ARTIFACTS_DIR.as_uri(),
        )
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, y_train, X_test, y_test = load_processed_data()

    # 1. Modelo Baseline: Ridge
    ridge_params = {"alpha": 1.0, "random_state": 42}
    ridge_model = Ridge(**ridge_params)
    train_and_log_model(
        model_name="Ridge_Regression_Baseline",
        model_instance=ridge_model,
        params=ridge_params,
        X_train=X_train,
        y_train_log=y_train,
        X_test=X_test,
        y_test_log=y_test,
    )

    # 2. Modelo Ensamblado: Random Forest Regressor
    rf_params = {
        "n_estimators": 150,
        "max_depth": 12,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42,
        "n_jobs": -1,
    }
    rf_model = RandomForestRegressor(**rf_params)
    train_and_log_model(
        model_name="Random_Forest_Regressor",
        model_instance=rf_model,
        params=rf_params,
        X_train=X_train,
        y_train_log=y_train,
        X_test=X_test,
        y_test_log=y_test,
    )

    print("\n[Éxito] Pipeline de experimentación completado exitosamente.")


if __name__ == "__main__":
    run_experiment_pipeline()