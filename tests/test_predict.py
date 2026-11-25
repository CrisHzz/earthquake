"""Inferencia end-to-end: entrena un modelo chico, lo guarda, predice por código y por la API."""

import io

import joblib
import pytest
from fastapi.testclient import TestClient

import predict
from clasificacion_fuerte import crear_etiqueta, crear_pipeline_completo
from preprocesamiento import dividir_datos


@pytest.fixture(scope="module")
def modelo_chico(datos_muestra, ruta_limites):
    X_train, _, _, y_train, _, _ = dividir_datos(datos_muestra)
    modelo = crear_pipeline_completo()
    modelo.set_params(
        preprocesamiento__peso_zona__boundaries_path=ruta_limites,
        modelo__n_estimators=20,
        modelo__max_depth=5,
        modelo__n_jobs=1,
    )
    modelo.fit(X_train, crear_etiqueta(y_train))

    # Ida y vuelta por joblib (en memoria) para verificar que el pipeline se serializa.
    buffer = io.BytesIO()
    joblib.dump(modelo, buffer)
    buffer.seek(0)
    return joblib.load(buffer)


@pytest.fixture
def modelo_cargado(modelo_chico, monkeypatch):
    # Usa el modelo chico del test en lugar del modelo real de models/.
    monkeypatch.setattr(predict, "_modelo", modelo_chico)


def test_predecir_devuelve_probabilidad_valida(modelo_cargado):
    resultado = predict.predecir(-33.0, -71.6, 30.0)

    assert 0.0 <= resultado["probabilidad_fuerte"] <= 1.0
    assert resultado["es_fuerte"] in (0, 1)
    assert resultado["es_fuerte"] == int(resultado["probabilidad_fuerte"] >= resultado["umbral"])


def test_api_predict_end_to_end(modelo_cargado):
    from app.api import app

    cliente = TestClient(app)
    assert cliente.get("/health").json() == {"status": "ok"}

    respuesta = cliente.post(
        "/predict", json={"latitude": 38.3, "longitude": 142.4, "depth": 30}
    )
    assert respuesta.status_code == 200
    datos = respuesta.json()
    assert 0.0 <= datos["probabilidad_fuerte"] <= 1.0
    assert datos["es_fuerte"] in (0, 1)


def test_api_rechaza_entrada_invalida(modelo_cargado):
    from app.api import app

    cliente = TestClient(app)
    respuesta = cliente.post(
        "/predict", json={"latitude": 200, "longitude": 0, "depth": 10}
    )
    assert respuesta.status_code == 422
