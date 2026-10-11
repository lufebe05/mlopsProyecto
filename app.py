"""
Servicio Web FastAPI con renderizado Jinja2 para predicción de precios.
"""

from pathlib import Path
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from src.models.predict_model import CarPricePredictor

app = FastAPI(title="Car Price Prediction App", version="1.0.0")

# Configurar directorio de templates
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Cargar el predictor entrenado
predictor = CarPricePredictor()


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """Página principal con formulario de captura."""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"request": request, "prediction": None, "input_data": None},
    )


@app.post("/predict", response_class=HTMLResponse)
def predict_price(
    request: Request,
    name: str = Form(...),
    year: int = Form(...),
    km_driven: int = Form(...),
    fuel: str = Form(...),
    seller_type: str = Form(...),
    transmission: str = Form(...),
    owner: str = Form(...),
):
    """Recibe los datos del formulario, ejecuta el modelo y muestra la estimación."""
    vehicle_payload = {
        "name": name,
        "year": year,
        "km_driven": km_driven,
        "fuel": fuel,
        "seller_type": seller_type,
        "transmission": transmission,
        "owner": owner,
    }

    prediction_result = predictor.predict(vehicle_payload)[0]
    formatted_prediction = f"{prediction_result:,.2f}"

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request,
            "prediction": formatted_prediction,
            "input_data": vehicle_payload,
        },
    )


@app.get("/health")
def health():
    """Endpoint de monitoreo y diagnóstico de salud."""
    return {"status": "healthy", "service": "Car Price Predictor API"}


if __name__ == "__main__":
    import uvicorn

    print("--- Iniciando Car Price Predictor App ---")
    print("📍 Abre tu navegador en: http://127.0.0.1:8000")
    print("⏹️  Para detener: Ctrl + C")
    uvicorn.run(app, host="127.0.0.1", port=8000)