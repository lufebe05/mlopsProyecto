"""
Script de adquisición y descarga reproducible de datos.
Descarga el dataset original de CarDekho desde el repositorio público de referencia
y lo almacena localmente como data/raw/carData.csv.
"""

from pathlib import Path
import urllib.request
import pandas as pd

# Enlace raw directo al archivo CSV en GitHub
DATA_RAW_URL = (
    "https://raw.githubusercontent.com/aashok30/Car-Data-Analysis/"
    "main/CAR%20DETAILS%20FROM%20CAR%20DEKHO.csv"
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
TARGET_FILE = RAW_DATA_DIR / "carData.csv"

EXPECTED_COLUMNS = [
    "name",
    "year",
    "selling_price",
    "km_driven",
    "fuel",
    "seller_type",
    "transmission",
    "owner",
]


def download_dataset(force: bool = False) -> Path:
    """
    Descarga y valida el archivo carData.csv en data/raw/.
    
    Args:
        force (bool): Si es True, vuelve a descargar el archivo aunque ya exista.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if TARGET_FILE.exists() and not force:
        print(f"[INFO] El dataset ya existe localmente en: {TARGET_FILE}")
    else:
        print(f"[INFO] Descargando dataset desde la fuente pública...")
        try:
            urllib.request.urlretrieve(DATA_RAW_URL, TARGET_FILE)
            print(f"[OK] Archivo descargado exitosamente como: {TARGET_FILE.name}")
        except Exception as e:
            print(f"[ERROR] Error al descargar el archivo: {e}")
            raise e

    # Validación de integridad y lectura
    df = pd.read_csv(TARGET_FILE)
    print(f"[VALIDACIÓN] Dataset verificado: {df.shape[0]} filas y {df.shape[1]} columnas.")

    # Validar que las columnas esperadas estén presentes
    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"[ERROR] Faltan columnas en el dataset descargado: {missing_cols}")

    print("[VALIDACIÓN] Columnas requeridas presentes y validadas.")
    return TARGET_FILE


if __name__ == "__main__":
    download_dataset()