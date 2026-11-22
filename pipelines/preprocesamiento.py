import sys
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Permite importar los módulos de src/ desde cualquier directorio
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from zona_sismica import PesoZonaSismica  # noqa: E402

SEMILLA = 42
VARIABLE_OBJETIVO = "Magnitude"
ENTRADAS_ORIGINALES = ["Latitude", "Longitude", "Depth"]
COLUMNAS_MODELO = ENTRADAS_ORIGINALES + ["peso_zona_sismica"]


def cargar_datos(ruta=PROJECT_ROOT / "data" / "database.csv"):
    return pd.read_csv(ruta)


def dividir_datos(df):
    # Partición 70/20/10: entrenamiento, prueba, validación
    X = df.drop(columns=VARIABLE_OBJETIVO)
    y = df[VARIABLE_OBJETIVO]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=SEMILLA
    )
    X_test, X_val, y_test, y_val = train_test_split(
        X_temp, y_temp, test_size=1 / 3, random_state=SEMILLA
    )
    return X_train, X_test, X_val, y_train, y_test, y_val


def crear_pipeline_preprocesamiento():
    imputar_entradas = ColumnTransformer(
        [("imputar", SimpleImputer(strategy="median"), ENTRADAS_ORIGINALES)],
        remainder="drop",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")

    preprocesamiento = ColumnTransformer(
        [
            (
                "numericas",
                Pipeline(
                    [
                        ("imputar", SimpleImputer(strategy="median")),
                        ("escalar", StandardScaler()),
                    ]
                ),
                COLUMNAS_MODELO,
            )
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")

    return Pipeline(
        [
            ("imputar_entradas", imputar_entradas),
            ("peso_zona", PesoZonaSismica(cell_size_deg=2, random_state=SEMILLA)),
            ("preprocesamiento", preprocesamiento),
        ]
    )


def main():
    df = cargar_datos()
    X_train, X_test, X_val, y_train, y_test, y_val = dividir_datos(df)

    # Ajuste solo con entrenamiento para evitar filtración de datos
    pipeline = crear_pipeline_preprocesamiento()
    X_train_prep = pipeline.fit_transform(X_train, y_train)
    X_test_prep = pipeline.transform(X_test)
    X_val_prep = pipeline.transform(X_val)

    print(f"Entrenamiento: {X_train_prep.shape}")
    print(f"Prueba: {X_test_prep.shape}")
    print(f"Validación: {X_val_prep.shape}")
    return pipeline, X_train_prep, X_test_prep, X_val_prep


if __name__ == "__main__":
    main()
