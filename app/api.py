"""API FastAPI del Earthquake Estimator. Delgada: solo valida y llama a src/predict.py."""

import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR.parent / "src"))

from predict import predecir  # noqa: E402

app = FastAPI(
    title="Earthquake Estimator",
    description=(
        "Estima la probabilidad de que un terremoto sea fuerte (Magnitude >= 6.5) "
        "a partir de su latitud, longitud y profundidad. Proyecto académico: "
        "no es una herramienta de alerta temprana."
    ),
    version="0.1.0",
)


class EventoEntrada(BaseModel):
    latitude: float = Field(..., ge=-90, le=90, description="Latitud en grados", examples=[-33.0])
    longitude: float = Field(..., ge=-180, le=180, description="Longitud en grados", examples=[-71.6])
    depth: float = Field(..., ge=0, le=800, description="Profundidad en km", examples=[30])


class PrediccionSalida(BaseModel):
    probabilidad_fuerte: float = Field(..., description="Probabilidad de Magnitude >= 6.5")
    es_fuerte: int = Field(..., description="1 = fuerte, 0 = no fuerte (umbral 0.5)")
    umbral: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PrediccionSalida)
def predict(evento: EventoEntrada):
    try:
        return predecir(evento.latitude, evento.longitude, evento.depth)
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error))


app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/", include_in_schema=False)
def inicio():
    return FileResponse(BASE_DIR / "static" / "index.html")