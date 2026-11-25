# Proyecto 1: Earthquake Estimator
### Especialización en Ciencia de Datos e Inteligencia Artificial

**Integrantes:**
- Cristian Camilo Hernández Gutiérrez.
- Juan Sebastián Arango Muriel.
- Sara Medina Molina.

---

## ¿Qué resuelve?

Los terremotos son fenómenos naturales que pueden generar consecuencias importantes para las poblaciones, las infraestructuras y los ecosistemas. El análisis de los registros sísmicos históricos permite identificar patrones y comprender mejor las características asociadas a estos eventos.

**Pregunta del proyecto:** ¿Es posible estimar si un terremoto será fuerte (magnitud de 6.5 o superior) a partir de su ubicación y profundidad, utilizando un modelo de aprendizaje automático interpretable?

**Objetivo:** Desarrollar un modelo de clasificación basado en árboles de decisión que estime la probabilidad de que un terremoto sea fuerte (`Magnitude >= 6.5`) a partir de su latitud, longitud y profundidad. Se busca priorizar la interpretabilidad del modelo, de manera que sea posible comprender qué características tienen mayor influencia en las predicciones.

**Dataset:** [Earthquake Database – USGS (Kaggle)](https://www.kaggle.com/datasets/usgs/earthquake-database).

El conjunto de datos contiene registros históricos de terremotos con magnitudes reportadas de 5.5 o superiores desde 1965. Incluye información sobre fecha, hora, ubicación geográfica, profundidad, magnitud y fuentes de registro.

**Modelo oficial:** `RandomForestClassifier` (`pipelines/clasificacion_fuerte.py`), guardado como pipeline completo (preprocesamiento + clasificador) en `models/modelo_clasificador_fuerte.joblib`, versionado con DVC y registrado con GTO como el artefacto `clasificador-sismo-fuerte`.

- **Entradas:** `Latitude`, `Longitude` y `Depth`. El pipeline calcula internamente la variable `peso_zona_sismica` (60 % actividad histórica de la zona + 40 % distancia al límite de placa tectónica).
- **Salida:** la probabilidad de que el terremoto sea fuerte (`predict_proba`) y la etiqueta 1 = fuerte / 0 = no fuerte (`predict`, umbral por defecto de 0.5).

El modelo se evalúa con métricas para clases desbalanceadas (cerca del 10 % de los eventos son fuertes): ROC_AUC, PR_AUC, F1, precisión, recall y exactitud balanceada. Los hiperparámetros se optimizan con ROC_AUC (validación cruzada estratificada de 5 folds sobre entrenamiento).

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

**Variable objetivo:** `fuerte`, derivada de `Magnitude` (1 si `Magnitude >= 6.5`, 0 en otro caso).

Las variables predictoras definitivas se establecerán durante el análisis exploratorio y la selección de características, considerando su relevancia, calidad y disponibilidad, evitando la fuga de información (*data leakage*).

---

## Estructura del proyecto

El repositorio sigue una arquitectura modular inspirada en la plantilla de referencia del curso. Esta organización separa la experimentación, el procesamiento de datos, el entrenamiento, la predicción y la presentación de resultados.

```text
earthquake/
│
├── README.md
├── Makefile
├── pyproject.toml
├── uv.lock
├── main.py
├── dvc.yaml                 # Registro de artefactos (model registry de DVC/GTO)
├── .dvcignore
├── .python-version
│
├── notebooks/
│   ├── 01_exploracion_datos.ipynb
│   └── 02_preprocesamiento.ipynb
│
├── src/
│   ├── download_data.py     # Descarga el dataset de terremotos desde Kaggle
│   ├── download_pb2002.py   # Descarga los límites de placa PB2002
│   ├── zona_sismica.py      # Calcula el peso de zona sísmica (feature geográfica)
│   └── predict.py           # Carga el modelo y realiza predicciones
│
├── pipelines/
│   ├── clasificacion_fuerte.py  # Entrena y versiona el modelo oficial
│   ├── preprocesamiento.py      # Limpieza y preparación de los datos
│   ├── evaluar.py               # Evalúa el modelo guardado sobre el conjunto de prueba
│   └── pruebas_peso_zona.py     # Pruebas de la feature peso_zona_sismica
│
├── app/
│   ├── api.py               # API FastAPI (valida y delega en src/predict.py)
│   └── static/
│       └── index.html       # Interfaz web del estimador
│
├── data/
│   ├── database.csv.dvc         # Puntero DVC del dataset de terremotos
│   └── PB2002_steps.csv.dvc     # Puntero DVC de los límites de placa
│
├── models/
│   └── modelo_clasificador_fuerte.joblib.dvc  # Puntero DVC del modelo
│
└── tests/
```

### ¿Qué hace cada carpeta?

| Carpeta/archivo | Responsabilidad |
|---|---|
| `README.md` | Documentación general del proyecto, instrucciones y resultados. |
| `Makefile` | Centraliza los comandos para configurar, entrenar, probar y ejecutar el proyecto. |
| `pyproject.toml` | Define las dependencias y configuración del proyecto. |
| `uv.lock` | Mantiene versiones reproducibles de las dependencias. |
| `dvc.yaml` | Registra el modelo como artefacto versionado (model registry de DVC/GTO). |
| `notebooks/` | Experimentación, análisis exploratorio y preprocesamiento. |
| `src/` | Código fuente reutilizable: descarga de datos, cálculo de features y predicción. |
| `pipelines/` | Scripts ejecutables de preprocesamiento, entrenamiento y evaluación. |
| `app/` | API FastAPI e interfaz web para consultar predicciones. |
| `tests/` | Pruebas automatizadas para verificar el funcionamiento del proyecto. |
| `models/` | Almacena el modelo entrenado y serializado (versionado con DVC). |
| `data/` | Almacena los datos descargados o generados localmente (versionados con DVC). |

### ¿Qué hace cada módulo en `src/` y `pipelines/`?

| Archivo | Responsabilidad |
|---|---|
| `src/download_data.py` | Descarga el dataset de terremotos de Kaggle a `data/database.csv`. |
| `src/download_pb2002.py` | Descarga los tramos de límites de placa PB2002 a `data/PB2002_steps.csv`. |
| `src/zona_sismica.py` | Calcula la feature `peso_zona_sismica` a partir de la ubicación y los límites de placa. |
| `src/predict.py` | Carga el modelo entrenado y realiza predicciones sobre nuevos eventos. |
| `pipelines/preprocesamiento.py` | Limpieza y preparación de los datos para el entrenamiento. |
| `pipelines/clasificacion_fuerte.py` | Entrena el `RandomForestClassifier` y guarda el pipeline completo. |
| `pipelines/evaluar.py` | Evalúa el modelo guardado sobre el conjunto de prueba sin reentrenar. |
| `pipelines/pruebas_peso_zona.py` | Pruebas de importancia y aporte de la feature `peso_zona_sismica`. |

El principio de diseño es mantener una responsabilidad específica por módulo, evitando duplicar la lógica entre el entrenamiento y la aplicación.

---

## Cómo ejecutar el proyecto

El proyecto utiliza Python y `uv` para gestionar las dependencias y mantener un entorno reproducible. El `Makefile` centraliza la inicialización para que cualquier integrante deje el entorno listo con un solo comando.

### 1. Clonar el repositorio

```bash
git clone <URL_DEL_REPOSITORIO>
cd earthquake
```

### 2. Inicializar el proyecto

La forma recomendada es usar el `Makefile`, que instala las dependencias y registra el kernel de Jupyter para empezar a trabajar de inmediato.

**En Windows** (requiere tener `uv` instalado; `make` se instala con `choco install make` o `scoop install make`):

```bash
make setup
```

**En Linux (Debian/Ubuntu):** instala `uv` automáticamente si no está presente y luego hace el setup completo:

```bash
make setup-linux
```

> Tras `make setup-linux`, si la terminal no reconoce `uv`, abre una nueva o ejecuta `source $HOME/.local/bin/env`.

Ambos comandos dejan el entorno listo y registran un kernel de Jupyter llamado `earthquake`, que debes seleccionar al abrir los notebooks.

### 2b. Alternativa: instalación tradicional con uv

Si prefieres no usar `make`, puedes replicar el setup manualmente:

```bash
# Instala/sincroniza las dependencias en el entorno .venv
uv sync

# (Opcional) Registra el entorno como kernel de Jupyter
uv run python -m ipykernel install --user --name earthquake --display-name "earthquake"
```

#### Comandos disponibles en el `Makefile`

| Comando | Descripción |
|---|---|
| `make setup` | Setup completo en Windows/Mac/Linux (requiere `uv` ya instalado). |
| `make setup-linux` | Setup en Debian/Ubuntu: instala `uv` si falta + setup completo. |
| `make install` | Solo sincroniza las dependencias (`uv sync`). |
| `make kernel` | Registra el entorno como kernel de Jupyter. |
| `make run` | Ejecuta `main.py` dentro del entorno. |
| `make api` | Levanta la API de FastAPI con `uvicorn` y sirve la página web en http://127.0.0.1:8000. |
| `make clean` | Elimina el entorno virtual y las cachés. |
| `make help` | Lista todos los comandos disponibles. |

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
make api
```

Este comando levanta la API de FastAPI con `uvicorn` y sirve la página web en el puerto `8000`. Una vez iniciado el servidor, abre en el navegador:

- **Interfaz web:** http://127.0.0.1:8000/
- **Documentación interactiva (Swagger):** http://127.0.0.1:8000/docs

La aplicación permite ingresar las características de un evento sísmico (latitud, longitud y profundidad) y consultar la probabilidad de que sea un terremoto fuerte según el modelo.

---

## Flujo completo desde cero

Si acabas de clonar el repositorio y no tienes nada preparado (ni entorno, ni datos, ni modelo), estos son los pasos en orden hasta dejar la aplicación funcionando:

```bash
# 1. Instalar dependencias y dejar el entorno listo
make setup          # en Linux/Debian/Ubuntu usa: make setup-linux

# 2. Descargar el dataset de terremotos desde Kaggle a data/database.csv
make data

# 3. Entrenar el modelo, guardarlo en models/ y versionarlo (DVC + GTO)
make train

# 4. Levantar la API y servir la página web en http://127.0.0.1:8000
make api
```

En una sola línea (cada paso solo continúa si el anterior termina bien):

```bash
make setup && make data && make train && make api
```

Al terminar, abre http://127.0.0.1:8000/ en el navegador para usar la interfaz.

> **Nota:** `make data` descarga el dataset desde Kaggle y requiere credenciales de la API de Kaggle (variable de entorno `KAGGLE_USERNAME`/`KAGGLE_KEY` o el archivo `kaggle.json`). Si el modelo ya está versionado con DVC, puedes traerlo directamente con `make pull-data` en lugar de reentrenar.

---

## Evaluación del modelo

El rendimiento del clasificador se evalúa con métricas para clases desbalanceadas:

| Métrica | Propósito |
|---|---|
| ROC_AUC | Capacidad de ordenar los eventos fuertes por encima de los no fuertes; métrica principal, usada para optimizar los hiperparámetros (objetivo mínimo 0.6). |
| PR_AUC | Área bajo la curva precisión-recall con la clase fuerte minoritaria. |
| F1, precisión, recall | Calidad de la etiqueta con el umbral por defecto. |
| Exactitud balanceada | Promedio del recall de ambas clases. |

Resultados en el conjunto de prueba (evaluado una sola vez):

| Modelo | ROC_AUC | PR_AUC |
|---|---|---|
| Línea base (`DummyClassifier`) | 0.4952 | 0.0939 |
| Random Forest | 0.6415 | 0.1552 |

Las pruebas sobre `peso_zona_sismica` (`pipelines/pruebas_peso_zona.py`) usan solo entrenamiento y validación: importancia por permutación de las 4 variables con PR_AUC y comparación del clasificador con y sin el peso de zona.

Ejecución desde la raíz del proyecto:

```bash
uv run python pipelines/clasificacion_fuerte.py
uv run python pipelines/pruebas_peso_zona.py
```

---


## Versionado de datos y modelo (DVC + GTO)

El proyecto no guarda los archivos grandes (datasets y modelo) en Git. En su lugar usa **DVC** para versionar los datos y el modelo, y **GTO** como registro de versiones del modelo. Git solo guarda archivos de puntero `.dvc` (texto ligero con el hash del contenido) y el `dvc.yaml` con el registro de artefactos.

### Datasets versionados con DVC

Cada dataset se versiona con un archivo `.dvc` que guarda su hash MD5, tamaño y nombre. El contenido real se almacena por hash en el cache de DVC (`.dvc/cache/`) y en el remoto configurado.

| Dataset | Puntero DVC (en Git) | Hash MD5 | Tamaño |
|---|---|---|---|
| `data/database.csv` | `data/database.csv.dvc` | `6b72dda0...` | ~2.4 MB |
| `data/PB2002_steps.csv` | `data/PB2002_steps.csv.dvc` | `f2678aab...` | ~238 KB |

Ejemplo del contenido de `data/database.csv.dvc`:

```yaml
outs:
- md5: 6b72dda08179509da5b48ebeeb2a2e9f
  size: 2397103
  hash: md5
  path: database.csv
```

**Remoto DVC** (configurado en `.dvc/config`): remoto llamado `local` que apunta a la carpeta `dvc-remote/` del proyecto.

```ini
[core]
    remote = local
['remote "local"']
    url = <ruta>/dvc-remote
```

Comandos útiles:

```bash
# Traer los datos versionados (según su .dvc)
uv run dvc pull data/database.csv.dvc data/PB2002_steps.csv.dvc   # o: make pull-data

# Ver el estado de los archivos versionados
uv run dvc status

# Listar los remotos configurados
uv run dvc remote list
```

### Modelo versionado con DVC

El modelo entrenado también se versiona con DVC (no se guarda en Git):

| Modelo | Puntero DVC (en Git) | Hash MD5 | Tamaño |
|---|---|---|---|
| `models/modelo_clasificador_fuerte.joblib` | `models/modelo_clasificador_fuerte.joblib.dvc` | `ce44585e...` | ~10.5 MB |

```bash
# Traer el modelo versionado sin reentrenar
uv run dvc pull models/modelo_clasificador_fuerte.joblib.dvc
```

### Model registry (DVC + GTO)

El modelo está registrado como **artefacto** en `dvc.yaml`. Este registro es el *model registry*: le da un nombre estable (`clasificador-sismo-fuerte`), un tipo, una descripción y etiquetas, de modo que pueda listarse y versionarse.

```yaml
artifacts:
  clasificador-sismo-fuerte:
    path: models/modelo_clasificador_fuerte.joblib
    type: model
    desc: Pipeline completo (preprocesamiento + RandomForestClassifier) que estima
          la probabilidad de Magnitude >= 6.5
    labels:
      - random-forest
      - clasificacion
```

Durante `make train`, además de versionar el modelo con DVC, se registra una nueva versión con **GTO** (`gto register clasificador-sismo-fuerte`). Para inspeccionar el registro:

```bash
# Listar los artefactos registrados
uv run dvc artifacts list .

# Ver el historial de versiones registradas con GTO
uv run gto history
```

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
| Matplotlib / Seaborn / Plotly | Visualización y análisis exploratorio. |
| Optuna | Optimización de hiperparámetros. |
| FastAPI / Uvicorn | API REST y servidor de la aplicación web. |
| DVC | Versionado de datos y modelos (data version control). |
| GTO | Registro y etiquetado de versiones del modelo (model registry). |
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
