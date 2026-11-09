# Proyecto 1: Earthquake Estimator
### Especialización en Ciencia de Datos e Inteligencia Artificial

**Integrantes:**
- Cristian Camilo Hernández Gutiérrez.
- Juan Sebastián Arango Muriel.
- Sara Medina Molina.

---

## ¿Qué resuelve?

Los terremotos son fenómenos naturales que pueden generar consecuencias importantes para las poblaciones, las infraestructuras y los ecosistemas. El análisis de los registros sísmicos históricos permite identificar patrones y comprender mejor las características asociadas a estos eventos.

**Pregunta del proyecto:** ¿Es posible estimar la magnitud de un terremoto a partir de sus características geográficas, temporales y sísmicas, utilizando un modelo de aprendizaje automático interpretable?

**Objetivo:** Desarrollar un modelo de regresión basado en árboles de decisión que permita estimar la magnitud de los terremotos a partir de variables relevantes de sus registros históricos. Se busca priorizar la interpretabilidad del modelo, de manera que sea posible comprender qué características tienen mayor influencia en las predicciones y explicar los criterios utilizados para generar cada estimación.

**Dataset:** [Earthquake Database – USGS (Kaggle)](https://www.kaggle.com/datasets/usgs/earthquake-database).

El conjunto de datos contiene registros históricos de terremotos con magnitudes reportadas de 5.5 o superiores desde 1965. Incluye información sobre fecha, hora, ubicación geográfica, profundidad, magnitud y fuentes de registro.

**Modelo:** Se explorará inicialmente un modelo de árbol de decisión para regresión (`DecisionTreeRegressor`), considerando su capacidad para representar relaciones no lineales y facilitar la interpretación de los resultados.

El modelo será evaluado mediante métricas apropiadas para problemas de regresión, como MAE, RMSE y R².

---

## Dataset y variables

El conjunto de datos contiene 21 columnas que describen diferentes características de los eventos sísmicos.

| Variable | Descripción |
|---|---|
| `Date` | Fecha en la que ocurrió el terremoto. |
| `Time` | Hora del evento sísmico. |
| `Latitude` | Latitud geográfica del epicentro. |
| `Longitude` | Longitud geográfica del epicentro. |
| `Type` | Tipo de evento sísmico. |
| `Depth` | Profundidad del terremoto. |
| `Depth Error` | Error asociado a la estimación de profundidad. |
| `Depth Seismic Stations` | Número de estaciones sísmicas utilizadas para determinar la profundidad. |
| `Magnitude` | Magnitud reportada del terremoto. |
| `Magnitude Type` | Tipo de escala de magnitud utilizada. |
| `Magnitude Error` | Error asociado a la estimación de magnitud. |
| `Magnitude Seismic Stations` | Número de estaciones utilizadas para determinar la magnitud. |
| `Azimuthal Gap` | Separación azimutal entre estaciones sísmicas. |
| `Horizontal Distance` | Distancia horizontal asociada a la localización. |
| `Horizontal Error` | Error asociado a la ubicación horizontal. |
| `Root Mean Square` | Medida del error RMS del ajuste de las señales sísmicas. |
| `ID` | Identificador del registro sísmico. |
| `Source` | Fuente del registro. |
| `Location Source` | Fuente utilizada para determinar la ubicación. |
| `Magnitude Source` | Fuente utilizada para determinar la magnitud. |
| `Status` | Estado del registro sísmico. |

**Variable objetivo:** `Magnitude`.

Las variables predictoras definitivas se establecerán durante el análisis exploratorio y la selección de características, considerando su relevancia, calidad y disponibilidad, evitando la fuga de información (*data leakage*).

---

## Estructura del proyecto

El repositorio sigue una arquitectura modular inspirada en la plantilla de referencia del curso. Esta organización separa la experimentación, el procesamiento de datos, el entrenamiento, la predicción y la presentación de resultados.

```text
earthquake-estimator/
│
├── README.md
├── Makefile
├── pyproject.toml
├── uv.lock
├── .gitignore
├── .python-version
│
├── notebooks/
│   └── 01_exploracion_datos.ipynb
│
├── src/
│   └── earthquake_estimator/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── features.py
│       ├── model.py
│       ├── train.py
│       └── predict.py
│
├── app/
│   └── streamlit_app.py
│
├── tests/
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_features.py
│   └── test_predict.py
│
├── models/
│   └── .gitkeep
│
├── data/
│   └── .gitkeep
│
├── reports/
│   └── .gitkeep
│
└── .github/
    └── workflows/
        └── ci.yml
```

*Los nombres de los módulos son iniciales y podrán modificarse a medida que avance el desarrollo.*

### ¿Qué hace cada carpeta?

| Carpeta/archivo | Responsabilidad |
|---|---|
| `README.md` | Documentación general del proyecto, instrucciones y resultados. |
| `Makefile` | Centraliza los comandos para configurar, entrenar, probar y ejecutar el proyecto. |
| `pyproject.toml` | Define las dependencias y configuración del proyecto. |
| `uv.lock` | Mantiene versiones reproducibles de las dependencias. |
| `notebooks/` | Experimentación, análisis exploratorio, visualizaciones y comparación de modelos. |
| `src/` | Contiene el código fuente modular y reutilizable. |
| `app/` | Interfaz para interactuar con el modelo y consultar predicciones. |
| `tests/` | Pruebas automatizadas para verificar el funcionamiento del proyecto. |
| `models/` | Almacena los modelos entrenados y serializados. |
| `data/` | Almacena los datos descargados o generados localmente. |
| `reports/` | Contiene métricas, gráficas y resultados del análisis. |
| `.github/workflows/` | Configuración de integración continua mediante GitHub Actions. |

### ¿Qué hace cada módulo en `src/`?

| Archivo | Responsabilidad |
|---|---|
| `config.py` | Centraliza las rutas, nombres de variables, parámetros y configuraciones. |
| `data.py` | Gestiona la descarga, carga y validación del conjunto de datos. |
| `features.py` | Realiza la limpieza, transformación y selección de características. |
| `model.py` | Define el pipeline de preprocesamiento y el modelo de regresión. |
| `train.py` | Ejecuta el entrenamiento, evaluación y almacenamiento del modelo. |
| `predict.py` | Carga el modelo entrenado y realiza predicciones sobre nuevos registros. |

El principio de diseño será mantener una responsabilidad específica por módulo, evitando duplicar la lógica entre el entrenamiento y la aplicación.

---

## Cómo ejecutar el proyecto

El proyecto utiliza Python y `uv` para gestionar las dependencias y mantener un entorno reproducible.

### 1. Clonar el repositorio

```bash
git clone <URL_DEL_REPOSITORIO>
cd earthquake-estimator
```

### 2. Instalar las dependencias

```bash
uv sync
```

### 3. Preparar los datos

```bash
make data
```

Este comando permitirá obtener y preparar el conjunto de datos desde su fuente original. Los datos crudos no se almacenarán directamente en GitHub.

### 4. Entrenar el modelo

```bash
make train
```

El proceso realizará el preprocesamiento, entrenamiento y evaluación del modelo, y almacenará el resultado en la carpeta `models/`.

### 5. Ejecutar la aplicación

```bash
make serve
```

La aplicación permitirá ingresar las características de un evento sísmico y consultar la magnitud estimada por el modelo.

*Los comandos anteriores representan la interfaz prevista del proyecto y se ajustarán a la configuración definitiva del Makefile.*

---

## Evaluación del modelo

El rendimiento del modelo se evaluará utilizando métricas de regresión:

| Métrica | Propósito |
|---|---|
| MAE | Mide el error absoluto promedio entre las magnitudes reales y estimadas. |
| RMSE | Penaliza en mayor medida los errores de predicción elevados. |
| R² | Evalúa qué proporción de la variabilidad de la variable objetivo explica el modelo. |

Además de las métricas, se analizará la interpretabilidad del árbol de decisión mediante:

- Importancia de las características (*Feature Importance*).
- Visualización de la estructura del árbol.
- Análisis de las reglas de decisión.
- Comparación entre valores reales y estimados.

Los resultados se documentarán una vez finalizado el entrenamiento y la evaluación.

---


## Integración continua (CI)

Se implementará un flujo de integración continua utilizando GitHub Actions, configurado en:

```text
.github/workflows/ci.yml
```

Este workflow permitirá automatizar:

- Configuración del entorno de Python.
- Instalación de dependencias.
- Verificación de calidad del código.
- Ejecución de pruebas automatizadas.

El objetivo es garantizar que los cambios realizados en el repositorio no afecten el funcionamiento del proyecto y que la rama evaluada mantenga todos los checks aprobados.

---

## Limitaciones del proyecto

El modelo tendrá las siguientes consideraciones:

- Los datos corresponden a terremotos con magnitud reportada de 5.5 o superior, por lo que no representan la totalidad de los eventos sísmicos.
- La información histórica no garantiza que los patrones identificados se mantengan en futuros eventos.
- La calidad de las predicciones dependerá de la precisión y disponibilidad de los registros originales.
- Un árbol de decisión puede presentar sobreajuste si su complejidad no se controla adecuadamente.
- Las estimaciones del modelo no constituyen predicciones de ocurrencia de terremotos ni permiten determinar cuándo o dónde sucederá uno.

El proyecto tiene un propósito académico y de análisis de datos. Sus resultados no deben utilizarse como herramienta de alerta temprana ni como sustituto de los sistemas oficiales de monitoreo sísmico.

---

## Tecnologías utilizadas

| Tecnología | Uso |
|---|---|
| Python | Lenguaje principal. |
| Pandas | Manipulación y análisis de datos. |
| NumPy | Operaciones numéricas. |
| Scikit-learn | Preprocesamiento, entrenamiento y evaluación del modelo. |
| Matplotlib / Seaborn | Visualización y análisis exploratorio. |
| Streamlit | Desarrollo de la aplicación interactiva. |
| Pytest | Pruebas automatizadas. |
| uv | Gestión de dependencias y entorno. |
| Git y GitHub | Control de versiones y colaboración. |
| GitHub Actions | Integración continua. |

---

## Autores

**Proyecto desarrollado por:**

- Cristian Camilo Hernández Gutiérrez.
- Juan Sebastián Arango Muriel.
- Sara Medina Molina.

**Especialización en Ciencia de Datos e Inteligencia Artificial.**

**Proyecto I.**
