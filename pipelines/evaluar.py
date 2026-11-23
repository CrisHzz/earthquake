import sys
from pathlib import Path

import joblib
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from clasificacion_fuerte import RUTA_MODELO, crear_etiqueta
from preprocesamiento import cargar_datos, dividir_datos


def main():
    # Evalúa el modelo ya guardado sobre el conjunto de prueba sin reentrenar
    if not RUTA_MODELO.exists():
        print(
            f"No se encontró el modelo en {RUTA_MODELO}.\n"
            "Ejecuta 'make train' antes de 'make test'."
        )
        sys.exit(1)

    df = cargar_datos()
    # Mismo split que el entrenamiento (SEMILLA=42) para reconstruir el test exacto
    _, X_test, _, _, y_test, _ = dividir_datos(df)
    y_test = crear_etiqueta(y_test)

    modelo = joblib.load(RUTA_MODELO)
    prob = modelo.predict_proba(X_test)[:, 1]
    pred = modelo.predict(X_test)

    print(f"Modelo cargado desde {RUTA_MODELO}")
    print("\nPrueba:")
    print(
        f"  Random Forest  ROC_AUC={roc_auc_score(y_test, prob):.4f}"
        f"  PR_AUC={average_precision_score(y_test, prob):.4f}"
        f"  F1={f1_score(y_test, pred, zero_division=0):.4f}"
        f"  Precision={precision_score(y_test, pred, zero_division=0):.4f}"
        f"  Recall={recall_score(y_test, pred, zero_division=0):.4f}"
        f"  BalAcc={balanced_accuracy_score(y_test, pred):.4f}"
    )


if __name__ == "__main__":
    main()
