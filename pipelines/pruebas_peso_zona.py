import joblib
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from clasificacion_fuerte import RUTA_MODELO, crear_etiqueta
from preprocesamiento import ENTRADAS_ORIGINALES, SEMILLA, cargar_datos, dividir_datos

# El conjunto de prueba no se usa aquí: ya se evaluó una vez en clasificacion_fuerte.py


def crear_pipeline_sin_peso(modelo_oficial):
    # Mismo clasificador y mismos hiperparámetros, solo con Latitude, Longitude y Depth
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
                ENTRADAS_ORIGINALES,
            )
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")
    return Pipeline(
        [
            ("preprocesamiento", preprocesamiento),
            ("modelo", clone(modelo_oficial.named_steps["modelo"])),
        ]
    )


def importancia_por_permutacion(modelo, X_val, y_val):
    # Se permutan las 4 variables que recibe el Random Forest (incluye peso_zona_sismica)
    X_val_prep = modelo.named_steps["preprocesamiento"].transform(X_val)
    resultado = permutation_importance(
        modelo.named_steps["modelo"],
        X_val_prep,
        y_val,
        scoring="average_precision",
        n_repeats=30,
        random_state=SEMILLA,
        n_jobs=1,
    )
    print("\nImportancia por permutación (validación, caída de PR_AUC, 30 repeticiones):")
    orden = resultado.importances_mean.argsort()[::-1]
    for i in orden:
        print(
            f"  {X_val_prep.columns[i]:<18} {resultado.importances_mean[i]:.4f}"
            f" ± {resultado.importances_std[i]:.4f}"
        )


def comparar_con_sin_peso(modelo_oficial, X_train, y_train, X_val, y_val):
    modelo_sin_peso = crear_pipeline_sin_peso(modelo_oficial).fit(X_train, y_train)
    modelos = {"Con peso zona": modelo_oficial, "Sin peso zona": modelo_sin_peso}

    print("\nComparación con / sin peso_zona_sismica (validación):")
    for nombre, modelo in modelos.items():
        prob = modelo.predict_proba(X_val)[:, 1]
        print(
            f"  {nombre:<14} ROC_AUC={roc_auc_score(y_val, prob):.4f}"
            f"  PR_AUC={average_precision_score(y_val, prob):.4f}"
        )

    # Validación cruzada estratificada de 5 folds solo sobre entrenamiento
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
    print("\nComparación con / sin peso_zona_sismica (CV 5 folds en entrenamiento):")
    for nombre, modelo in modelos.items():
        puntajes = cross_validate(
            clone(modelo), X_train, y_train, cv=cv, scoring=["roc_auc", "average_precision"]
        )
        roc, pr = puntajes["test_roc_auc"], puntajes["test_average_precision"]
        print(
            f"  {nombre:<14} ROC_AUC={roc.mean():.4f} ± {roc.std():.4f}"
            f"  PR_AUC={pr.mean():.4f} ± {pr.std():.4f}"
        )


def main():
    df = cargar_datos()
    X_train, _, X_val, y_train, _, y_val = dividir_datos(df)
    y_train, y_val = crear_etiqueta(y_train), crear_etiqueta(y_val)

    modelo_oficial = joblib.load(RUTA_MODELO)
    print(f"Modelo oficial cargado de {RUTA_MODELO}")

    importancia_por_permutacion(modelo_oficial, X_val, y_val)
    comparar_con_sin_peso(modelo_oficial, X_train, y_train, X_val, y_val)


if __name__ == "__main__":
    main()
