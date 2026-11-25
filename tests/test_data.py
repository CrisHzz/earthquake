"""Verifica la carga del dataset."""

from pathlib import Path

import preprocesamiento
from conftest import RUTA_MUESTRA

COLUMNAS_REQUERIDAS = {"Latitude", "Longitude", "Depth", "Magnitude"}


def test_carga_muestra_tiene_columnas_y_filas(datos_muestra):
    assert len(datos_muestra) == 2000
    assert COLUMNAS_REQUERIDAS.issubset(datos_muestra.columns)
    assert datos_muestra[list(COLUMNAS_REQUERIDAS)].notna().all().all()


def test_rangos_validos(datos_muestra):
    assert datos_muestra["Latitude"].between(-90, 90).all()
    assert datos_muestra["Longitude"].between(-180, 180).all()
    assert (datos_muestra["Depth"] >= -5).all()
    assert (datos_muestra["Magnitude"] >= 5.5).all()


def test_fallback_a_muestra_si_no_hay_dataset_completo(monkeypatch):
    monkeypatch.setattr(preprocesamiento, "RUTA_DATOS_COMPLETOS", Path("no_existe.csv"))
    monkeypatch.setattr(preprocesamiento, "RUTA_DATOS_MUESTRA", RUTA_MUESTRA)
    df = preprocesamiento.cargar_datos()
    assert len(df) == 2000


def test_division_70_20_10_sin_solapamiento(datos_muestra):
    X_train, X_test, X_val, *_ = preprocesamiento.dividir_datos(datos_muestra)
    assert len(X_train) + len(X_test) + len(X_val) == len(datos_muestra)
    assert abs(len(X_train) / len(datos_muestra) - 0.70) < 0.01
    assert set(X_train.index).isdisjoint(X_test.index)
    assert set(X_train.index).isdisjoint(X_val.index)
    assert set(X_test.index).isdisjoint(X_val.index)
