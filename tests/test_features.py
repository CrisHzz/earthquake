"""Verifica el pipeline de features (imputación + peso de zona sísmica + escalado)."""

import numpy as np
import pandas as pd

from preprocesamiento import COLUMNAS_MODELO, crear_pipeline_preprocesamiento, dividir_datos
from zona_sismica import PesoZonaSismica


def test_peso_zona_en_rango_1_a_100(datos_muestra, ruta_limites):
    X = datos_muestra[["Latitude", "Longitude"]]
    y = datos_muestra["Magnitude"]
    transformador = PesoZonaSismica(boundaries_path=ruta_limites)

    entrenado = transformador.fit_transform(X, y)
    nuevo = transformador.transform(X.head(50))

    for salida in (entrenado, nuevo):
        peso = salida["peso_zona_sismica"]
        assert peso.notna().all()
        assert peso.between(1, 100).all()


def test_pipeline_produce_columnas_del_modelo(datos_muestra, ruta_limites):
    X_train, X_test, _, y_train, _, _ = dividir_datos(datos_muestra)
    pipeline = crear_pipeline_preprocesamiento()
    pipeline.set_params(peso_zona__boundaries_path=ruta_limites)

    X_train_prep = pipeline.fit_transform(X_train, y_train)
    X_test_prep = pipeline.transform(X_test)

    assert list(X_train_prep.columns) == COLUMNAS_MODELO
    assert X_test_prep.shape == (len(X_test), len(COLUMNAS_MODELO))
    assert np.isfinite(X_test_prep.to_numpy()).all()


def test_pipeline_imputa_nulos_en_entradas(datos_muestra, ruta_limites):
    X_train, _, _, y_train, _, _ = dividir_datos(datos_muestra)
    pipeline = crear_pipeline_preprocesamiento()
    pipeline.set_params(peso_zona__boundaries_path=ruta_limites)
    pipeline.fit(X_train, y_train)

    con_nulos = pd.DataFrame(
        [{"Latitude": np.nan, "Longitude": -71.6, "Depth": np.nan}]
    )
    salida = pipeline.transform(con_nulos)
    assert np.isfinite(salida.to_numpy()).all()
