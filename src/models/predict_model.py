"""
Módulo de inferencia y predicción.
Carga el preprocesador y modelo serializados para estimar el precio
de venta en rupias indias (INR) a partir de datos crudos.
"""

from pathlib import Path
from typing import Dict, List, Union
import warnings
import joblib
import numpy as np
import pandas as pd

# Rutas de artefactos
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
MODEL_PATH = MODELS_DIR / "random_forest_tuned.joblib"

ORDINAL_OWNER_MAP = {
    "Test Drive Car": 0,
    "First Owner": 1,
    "Second Owner": 2,
    "Third Owner": 3,
    "Fourth & Above Owner": 4,
}


class CarPricePredictor:
    """Clase encapsuladora para inferencia en producción."""

    def __init__(self, preprocessor_path: Path = PREPROCESSOR_PATH, model_path: Path = MODEL_PATH):
        if not preprocessor_path.exists():
            raise FileNotFoundError(f"[Error] No se encontró el preprocesador en: {preprocessor_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"[Error] No se encontró el modelo en: {model_path}")

        self.preprocessor = joblib.load(preprocessor_path)
        self.model = joblib.load(model_path)
        self.feature_names = self.preprocessor.get_feature_names_out()

        # Extraer las marcas conocidas aprendidas por el OneHotEncoder
        cat_encoder = self.preprocessor.named_transformers_["cat"]
        # La columna 'brand' es el índice 3 en NOMINAL_FEATURES
        self.known_brands = set(cat_encoder.categories_[3])
        print("[Info] Artefactos cargados exitosamente (Preprocesador y Modelo).")

    def _transform_raw_features(self, df_raw: pd.DataFrame, reference_year: int = 2020) -> pd.DataFrame:
        """Aplica la ingeniería de variables base sobre datos crudos de entrada."""
        df = df_raw.copy()

        # 1. Extracción de marca y homologación con marcas conocidas
        df["brand"] = df["name"].apply(lambda x: str(x).split()[0].strip().title())
        df["brand"] = df["brand"].apply(lambda b: b if b in self.known_brands else "Other")

        # 2. Antigüedad del vehículo
        df["vehicle_age"] = reference_year - df["year"]

        # 3. Mapeo ordinal de propietarios
        df["owner"] = df["owner"].map(ORDINAL_OWNER_MAP).fillna(1).astype(int)

        # 4. Eliminar columnas que no van al ColumnTransformer
        df = df.drop(columns=["name", "year"])

        return df

    def predict(self, input_data: Union[Dict, List[Dict], pd.DataFrame]) -> np.ndarray:
        """
        Genera predicciones en la escala real (INR).
        Acepta un diccionario, lista de diccionarios o un DataFrame crudo.
        """
        if isinstance(input_data, dict):
            df_raw = pd.DataFrame([input_data])
        elif isinstance(input_data, list):
            df_raw = pd.DataFrame(input_data)
        elif isinstance(input_data, pd.DataFrame):
            df_raw = input_data.copy()
        else:
            raise ValueError("[Error] Formato de entrada no soportado. Use dict, list[dict] o DataFrame.")

        # Transformación estructural
        df_features = self._transform_raw_features(df_raw)

        # Transformación con nombres de columnas intactos
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X_array = self.preprocessor.transform(df_features)
            X_processed = pd.DataFrame(X_array, columns=self.feature_names)

            # Predicción en escala logarítmica
            y_pred_log = self.model.predict(X_processed)

        # Reversión a rupias indias originales
        return np.expm1(y_pred_log)


if __name__ == "__main__":
    predictor = CarPricePredictor()

    sample_vehicles = [
        {
            "name": "Maruti Swift Dzire VDI",
            "year": 2014,
            "km_driven": 65000,
            "fuel": "Diesel",
            "seller_type": "Individual",
            "transmission": "Manual",
            "owner": "First Owner",
        },
        {
            "name": "BMW X5 xDrive 30d",
            "year": 2018,
            "km_driven": 25000,
            "fuel": "Diesel",
            "seller_type": "Dealer",
            "transmission": "Automatic",
            "owner": "First Owner",
        },
    ]

    preds = predictor.predict(sample_vehicles)

    print("\n--- Inferencia de Prueba ---")
    for vehicle, pred in zip(sample_vehicles, preds):
        print(f"Vehículo: {vehicle['name']} ({vehicle['year']})")
        print(f"  Estimación de Precio: {pred:,.2f} INR\n")