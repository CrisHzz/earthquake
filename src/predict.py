"""Carga el modelo entrenado y realiza predicciones sobre nuevos eventos."""

import sys
from pathlib import Path

import joblib
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
RUTA_MODELO = PROJECT_ROOT / "models" / "modelo_clasificador_fuerte.joblib"
UMBRAL = 0.5

# El pipeline guardado referencia el módulo `zona_sismica` (src/):
# debe poder importarse antes de joblib.load.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

_modelo = None


def cargar_modelo(ruta=RUTA_MODELO):
    global _modelo
    if _modelo is None:
        ruta = Path(ruta)
        if not ruta.exists():
            raise FileNotFoundError(
                f"No se encontró el modelo en {ruta}. "
                "Ejecuta 'make train' o 'dvc pull models/modelo_clasificador_fuerte.joblib.dvc'."
            )
        _modelo = joblib.load(ruta)
    return _modelo


def predecir(latitud: float, longitud: float, profundidad: float) -> dict:
    """Devuelve la probabilidad de que el terremoto sea fuerte (Magnitude >= 6.5)."""
    modelo = cargar_modelo()
    X = pd.DataFrame(
        [{"Latitude": latitud, "Longitude": longitud, "Depth": profundidad}]
    )
    probabilidad = float(modelo.predict_proba(X)[0, 1])
    return {
        "probabilidad_fuerte": probabilidad,
        "es_fuerte": int(probabilidad >= UMBRAL),
        "umbral": UMBRAL,
    }