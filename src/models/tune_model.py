"""
Módulo de optimización de hiperparámetros con Optuna y MLflow.
Ejecuta búsqueda bayesiana para Random Forest minimizando el RMSE en escala INR.
"""

import logging
import os
from pathlib import Path
from typing import Tuple
import warnings
import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

# Silenciar salidas ruidosas
warnings.filterwarnings("ignore")
os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "false"
logging.getLogger("mlflow").setLevel(logging.ERROR)
optuna.logging.set_verbosity(optuna.logging.WARNING)

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

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=["selling_price_log"])
    y_train_log = train_df["selling_price_log"]

    X_test = test_df.drop(columns=["selling_price_log"])
    y_test_log = test_df["selling_price_log"]

    return X_train, y_train_log, X_test, y_test_log


def run_optuna_tuning(n_trials: int = 25):
    """Ejecuta el estudio bayesiano con Optuna y registra en MLflow."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    tracking_uri = f"sqlite:///{SQLITE_DB_PATH.as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, y_train_log, X_test, y_test_log = load_processed_data()

    print(f"[Info] Iniciando optimización con Optuna ({n_trials} pruebas)...")

    with mlflow.start_run(run_name="Random_Forest_Optuna_Study") as parent_run:
        mlflow.log_param("optimizer", "Optuna_TPE")
        mlflow.log_param("n_trials", n_trials)

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=50),
                "max_depth": trial.suggest_int("max_depth", 8, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4),
                "max_features": trial.suggest_float("max_features", 0.5, 1.0),
                "random_state": 42,
                "n_jobs": -1,
            }

            # Validación cruzada K-Fold (k=5)
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            cv_rmse_scores = []

            for train_idx, val_idx in kf.split(X_train):
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train_log.iloc[train_idx], y_train_log.iloc[val_idx]

                model = RandomForestRegressor(**params)
                model.fit(X_tr, y_tr)

                y_val_pred_log = model.predict(X_val)
                # Cálculo de RMSE en escala real INR
                y_val_real = np.expm1(y_val)
                y_val_pred_real = np.expm1(y_val_pred_log)

                rmse_fold = np.sqrt(mean_squared_error(y_val_real, y_val_pred_real))
                cv_rmse_scores.append(rmse_fold)

            mean_cv_rmse = float(np.mean(cv_rmse_scores))

            # Registrar trial como corrida hija en MLflow
            with mlflow.start_run(run_name=f"Trial_{trial.number}", nested=True):
                mlflow.log_params(params)
                mlflow.log_metric("cv_rmse_inr", mean_cv_rmse)

            return mean_cv_rmse

        # Estudio minimizando RMSE en INR
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params
        best_params["random_state"] = 42
        best_params["n_jobs"] = -1

        print("\n--- Mejor Configuración Encontrada ---")
        for k, v in study.best_params.items():
            print(f"  {k}: {v}")
        print(f"  Mejor CV RMSE (INR): {study.best_value:,.2f}")

        # Entrenar modelo final con todo X_train usando los mejores parámetros
        print("\n[Info] Entrenando modelo óptimo con datos completos de entrenamiento...")
        final_model = RandomForestRegressor(**best_params)
        final_model.fit(X_train, y_train_log)

        # Evaluación en partición de prueba (Test)
        y_test_pred_log = final_model.predict(X_test)
        y_test_real = np.expm1(y_test_log)
        y_test_pred_real = np.expm1(y_test_pred_log)

        final_rmse = float(np.sqrt(mean_squared_error(y_test_real, y_test_pred_real)))
        final_mae = float(mean_absolute_error(y_test_real, y_test_pred_real))
        final_r2 = float(r2_score(y_test_real, y_test_pred_real))

        # Registrar resultados en la corrida principal de MLflow
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_cv_rmse_inr", study.best_value)
        mlflow.log_metric("test_rmse_inr", final_rmse)
        mlflow.log_metric("test_mae_inr", final_mae)
        mlflow.log_metric("test_r2_score", final_r2)

        # Guardar artefactos
        best_model_path = MODELS_DIR / "random_forest_tuned.joblib"
        joblib.dump(final_model, best_model_path)

        mlflow.sklearn.log_model(
            sk_model=final_model,
            name="tuned_model",
            serialization_format="pickle",
        )
        mlflow.log_artifact(str(best_model_path), artifact_path="local_checkpoints")

        print("\n[Resultados Finales del Modelo Optimizado en Test]")
        print(f"  Test RMSE (INR): {final_rmse:,.2f}")
        print(f"  Test MAE (INR):  {final_mae:,.2f}")
        print(f"  Test R²:         {final_r2:.4f}")
        print(f"[Éxito] Modelo serializado guardado en: {best_model_path}")


if __name__ == "__main__":
    run_optuna_tuning(n_trials=25)