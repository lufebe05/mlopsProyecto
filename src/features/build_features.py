"""
Módulo de preprocesamiento e ingeniería de características.
Transforma los datos crudos aplicando las directrices técnicas del EDA
y genera las particiones procesadas para entrenamiento y evaluación.
"""

from pathlib import Path
from typing import Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler

# Rutas reproducibles
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "carData.csv"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Variables según análisis exploratorio
ORDINAL_OWNER_MAP = {
    "Test Drive Car": 0,
    "First Owner": 1,
    "Second Owner": 2,
    "Third Owner": 3,
    "Fourth & Above Owner": 4,
}

NOMINAL_FEATURES = ["fuel", "seller_type", "transmission", "brand"]
NUMERICAL_SCALED_FEATURES = ["km_driven", "vehicle_age"]
ORDINAL_FEATURES = ["owner"]


def load_raw_data(data_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Carga y valida la existencia del dataset crudo."""
    if not data_path.exists():
        raise FileNotFoundError(f"[Error] Archivo no encontrado en: {data_path}")
    print(f"[Info] Cargando datos crudos desde: {data_path.name}")
    return pd.read_csv(data_path)


def clean_and_extract_features(df: pd.DataFrame, reference_year: int = 2020) -> pd.DataFrame:
    """
    Aplica limpieza estructural y transformaciones iniciales:
    - Eliminación de registros duplicados exactos.
    - Extracción de la marca comercial (brand).
    - Agrupación de marcas con baja frecuencia (< 1% del total) en 'Other'.
    - Creación de antigüedad del vehículo (vehicle_age).
    - Mapeo ordinal para historial de propietarios.
    """
    initial_rows = len(df)
    df_cleaned = df.drop_duplicates().copy()
    dropped_duplicates = initial_rows - len(df_cleaned)
    print(f"[Info] Registros duplicados eliminados: {dropped_duplicates} (Filas restantes: {len(df_cleaned)})")

    # 1. Extracción de marca
    df_cleaned["brand"] = df_cleaned["name"].apply(lambda x: x.split()[0].strip().title())

    # 2. Agrupación de marcas minoritarias (< 1% del volumen)
    brand_proportions = df_cleaned["brand"].value_counts(normalize=True)
    rare_brands = brand_proportions[brand_proportions < 0.01].index.tolist()
    df_cleaned["brand"] = df_cleaned["brand"].apply(lambda b: "Other" if b in rare_brands else b)
    print(f"[Info] Marcas agrupadas como 'Other': {len(rare_brands)}")

    # 3. Antigüedad del vehículo
    df_cleaned["vehicle_age"] = reference_year - df_cleaned["year"]

    # 4. Mapeo ordinal de owner
    df_cleaned["owner"] = df_cleaned["owner"].map(ORDINAL_OWNER_MAP).fillna(1).astype(int)

    # 5. Descartar columnas base redundantes
    df_cleaned = df_cleaned.drop(columns=["name", "year"])

    return df_cleaned


def create_preprocessor() -> ColumnTransformer:
    """Construye el ColumnTransformer de scikit-learn para features numéricas y categóricas."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                RobustScaler(),
                NUMERICAL_SCALED_FEATURES,
            ),
            (
                "cat",
                OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
                NOMINAL_FEATURES,
            ),
            (
                "ord",
                "passthrough",
                ORDINAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def run_feature_engineering_pipeline(
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ejecuta el flujo completo de ingeniería de variables, separa train/test,
    ajusta el preprocesador exclusivamente sobre train y persiste artefactos.
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df_raw = load_raw_data()
    df_features = clean_and_extract_features(df_raw)

    # Separación en X (predictores) y y (target)
    X = df_features.drop(columns=["selling_price"])
    # Transformación logarítmica de la variable dependiente
    y = np.log1p(df_features["selling_price"])

    # Partición estratificada / determinista train-test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"[Info] Partición completada: Train ({len(X_train)} filas), Test ({len(X_test)} filas)")

    # Ajuste del transformador solo en entrenamiento (evita data leakage)
    preprocessor = create_preprocessor()
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    # Reconstrucción de DataFrames procesados
    train_processed = pd.DataFrame(X_train_transformed, columns=feature_names, index=X_train.index)
    train_processed["selling_price_log"] = y_train

    test_processed = pd.DataFrame(X_test_transformed, columns=feature_names, index=X_test.index)
    test_processed["selling_price_log"] = y_test

    # Guardado de datasets procesados
    train_output_path = PROCESSED_DATA_DIR / "train_processed.csv"
    test_output_path = PROCESSED_DATA_DIR / "test_processed.csv"
    preprocessor_output_path = MODELS_DIR / "preprocessor.joblib"

    train_processed.to_csv(train_output_path, index=False)
    test_processed.to_csv(test_output_path, index=False)
    joblib.dump(preprocessor, preprocessor_output_path)

    print(f"[Éxito] Dataset de entrenamiento guardado en: {train_output_path}")
    print(f"[Éxito] Dataset de prueba guardado en: {test_output_path}")
    print(f"[Éxito] Pipeline preprocesador serializado en: {preprocessor_output_path}")

    return train_processed, test_processed


if __name__ == "__main__":
    run_feature_engineering_pipeline()