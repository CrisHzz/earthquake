from pathlib import Path

import joblib
import numpy as np
from scipy.stats import randint
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from preprocesamiento import (
    SEMILLA,
    cargar_datos,
    crear_pipeline_preprocesamiento,
    dividir_datos,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUTA_MODELO = PROJECT_ROOT / "models" / "modelo_random_forest.joblib"

ESPACIO_HIPERPARAMETROS = {
    "modelo__n_estimators": randint(100, 400),
    "modelo__max_depth": [None, 10, 20, 30],
    "modelo__min_samples_split": randint(2, 11),
    "modelo__min_samples_leaf": randint(1, 9),
    "modelo__max_features": ["sqrt", 0.5, 1.0],
}


def crear_pipeline_completo():
    # Preprocesamiento + modelo: se reajusta en cada fold, sin filtración de datos
    return Pipeline(
        [
            ("preprocesamiento", crear_pipeline_preprocesamiento()),
            ("modelo", RandomForestRegressor(random_state=SEMILLA, n_jobs=-1)),
        ]
    )


def calcular_metricas(y_real, y_pred):
    return {
        "MAE": mean_absolute_error(y_real, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_real, y_pred)),
        "R2": r2_score(y_real, y_pred),
    }


def imprimir_metricas(nombre_conjunto, modelos, X, y):
    print(f"\n{nombre_conjunto}:")
    for nombre, modelo in modelos.items():
        m = calcular_metricas(y, modelo.predict(X))
        print(
            f"  {nombre:<14} MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}  R2={m['R2']:.4f}"
        )


def main():
    df = cargar_datos()
    X_train, X_test, X_val, y_train, y_test, y_val = dividir_datos(df)

    # Sintonización con validación cruzada de 5 folds solo sobre entrenamiento
    busqueda = RandomizedSearchCV(
        crear_pipeline_completo(),
        ESPACIO_HIPERPARAMETROS,
        n_iter=10,
        cv=KFold(n_splits=5, shuffle=True, random_state=SEMILLA),
        scoring="neg_root_mean_squared_error",
        random_state=SEMILLA,
        n_jobs=1,
        verbose=1,
    )
    busqueda.fit(X_train, y_train)
    modelo = busqueda.best_estimator_

    print(f"Mejores hiperparámetros: {busqueda.best_params_}")
    print(f"RMSE CV (5 folds): {-busqueda.best_score_:.4f}")

    # Línea base: predice la media del entrenamiento
    linea_base = DummyRegressor(strategy="mean").fit(X_train, y_train)
    modelos = {"Línea base": linea_base, "Random Forest": modelo}

    imprimir_metricas("Validación", modelos, X_val, y_val)
    # El conjunto de prueba se evalúa una sola vez al final
    imprimir_metricas("Prueba", modelos, X_test, y_test)

    joblib.dump(modelo, RUTA_MODELO)
    print(f"\nModelo guardado en {RUTA_MODELO}")


if __name__ == "__main__":
    main()
