"""
Servicio Web FastAPI optimizado para contenedores Docker.
"""

from pathlib import Path
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from src.models.predict_model import CarPricePredictor

app = FastAPI(title="Car Price Prediction App (Docker)", version="1.0.0")

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

predictor = CarPricePredictor()


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index_docker.html",
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
        name="index_docker.html",
        context={
            "request": request,
            "prediction": formatted_prediction,
            "input_data": vehicle_payload,
        },
    )


@app.get("/health")
def health():
    return {"status": "healthy", "environment": "docker"}


if __name__ == "__main__":
    import uvicorn

    # host="0.0.0.0" permite recibir peticiones fuera del contenedor
    uvicorn.run(app, host="0.0.0.0", port=8000)