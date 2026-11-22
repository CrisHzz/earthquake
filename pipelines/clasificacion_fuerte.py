from scipy.stats import randint
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from preprocesamiento import (
    SEMILLA,
    cargar_datos,
    crear_pipeline_preprocesamiento,
    dividir_datos,
)

UMBRAL_FUERTE = 6.5

ESPACIO_HIPERPARAMETROS = {
    "modelo__n_estimators": randint(100, 400),
    "modelo__max_depth": [None, 10, 20, 30],
    "modelo__min_samples_split": randint(2, 11),
    "modelo__min_samples_leaf": randint(1, 9),
    "modelo__max_features": ["sqrt", 0.5, 1.0],
    "modelo__class_weight": ["balanced", "balanced_subsample", None],
}


def crear_etiqueta(magnitud):
    # 1 = terremoto fuerte (Magnitude >= 6.5), 0 = no fuerte
    return (magnitud >= UMBRAL_FUERTE).astype(int)


def crear_pipeline_completo():
    preprocesamiento = crear_pipeline_preprocesamiento()
    # y ya es la etiqueta 0/1: el peso de zona cuenta como fuerte y >= 1
    preprocesamiento.set_params(peso_zona__strong_threshold=1)
    return Pipeline(
        [
            ("preprocesamiento", preprocesamiento),
            ("modelo", RandomForestClassifier(random_state=SEMILLA, n_jobs=-1)),
        ]
    )


def imprimir_balance(nombre_conjunto, y):
    fuertes = int(y.sum())
    print(
        f"  {nombre_conjunto:<14} n={len(y)}  fuertes={fuertes} ({fuertes / len(y):.2%})"
        f"  no fuertes={len(y) - fuertes}"
    )


def imprimir_metricas(nombre_conjunto, modelos, X, y):
    print(f"\n{nombre_conjunto}:")
    for nombre, modelo in modelos.items():
        prob = modelo.predict_proba(X)[:, 1]
        pred = modelo.predict(X)
        print(
            f"  {nombre:<14} ROC_AUC={roc_auc_score(y, prob):.4f}"
            f"  PR_AUC={average_precision_score(y, prob):.4f}"
            f"  F1={f1_score(y, pred, zero_division=0):.4f}"
            f"  Precision={precision_score(y, pred, zero_division=0):.4f}"
            f"  Recall={recall_score(y, pred, zero_division=0):.4f}"
            f"  BalAcc={balanced_accuracy_score(y, pred):.4f}"
        )


def main():
    df = cargar_datos()
    X_train, X_test, X_val, y_train, y_test, y_val = dividir_datos(df)
    y_train, y_test, y_val = map(crear_etiqueta, (y_train, y_test, y_val))

    print("Balance de clases (Magnitude >= 6.5):")
    imprimir_balance("Entrenamiento", y_train)
    imprimir_balance("Validación", y_val)
    imprimir_balance("Prueba", y_test)

    # Sintonización con validación cruzada estratificada de 5 folds solo sobre entrenamiento
    busqueda = RandomizedSearchCV(
        crear_pipeline_completo(),
        ESPACIO_HIPERPARAMETROS,
        n_iter=10,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA),
        scoring="average_precision",
        random_state=SEMILLA,
        n_jobs=1,
        verbose=1,
    )
    busqueda.fit(X_train, y_train)
    modelo = busqueda.best_estimator_

    print(f"Mejores hiperparámetros: {busqueda.best_params_}")
    print(f"PR_AUC CV (5 folds): {busqueda.best_score_:.4f}")

    # Línea base: predice al azar respetando la proporción de clases del entrenamiento
    linea_base = DummyClassifier(strategy="stratified", random_state=SEMILLA).fit(
        X_train, y_train
    )
    modelos = {"Línea base": linea_base, "Random Forest": modelo}

    imprimir_metricas("Validación", modelos, X_val, y_val)
    # El conjunto de prueba se evalúa una sola vez al final
    imprimir_metricas("Prueba", modelos, X_test, y_test)


if __name__ == "__main__":
    main()
