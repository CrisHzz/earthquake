# Proyecto 1: Earthquake Estimator
### Especialización en Ciencia de Datos e Inteligencia Artificial

[![CI](https://github.com/CrisHzz/earthquake/actions/workflows/ci.yml/badge.svg)](https://github.com/CrisHzz/earthquake/actions/workflows/ci.yml)

**Integrantes:**
- Cristian Camilo Hernández Gutiérrez.
- Juan Sebastián Arango Muriel.
- Sara Medina Molina.

---

## ¿Qué resuelve?

Los terremotos son fenómenos naturales que pueden generar consecuencias importantes para las poblaciones, las infraestructuras y los ecosistemas. El análisis de los registros sísmicos históricos permite identificar patrones y comprender mejor las características asociadas a estos eventos.

**Pregunta del proyecto:** ¿Es posible estimar si un terremoto será fuerte (magnitud de 6.5 o superior) a partir de su ubicación y profundidad, utilizando un modelo de aprendizaje automático interpretable?

**Objetivo:** Desarrollar un modelo de clasificación basado en árboles de decisión que estime la probabilidad de que un terremoto sea fuerte (`Magnitude >= 6.5`) a partir de su latitud, longitud y profundidad. Se busca priorizar la interpretabilidad del modelo, de manera que sea posible comprender qué características tienen mayor influencia en las predicciones.

**Dataset:** [Earthquake Database – USGS (Kaggle)](https://www.kaggle.com/datasets/usgs/earthquake-database). Detalle de columnas, licencia, filas y filtros en [`dataset-card.md`](dataset-card.md).

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
├── dataset-card.md          # Documentación de los datasets
├── Makefile
├── pyproject.toml
├── uv.lock
├── main.py
├── dvc.yaml                 # Registro de artefactos (model registry de DVC/GTO)
├── .dvcignore
├── .python-version
│
├── .github/
│   └── workflows/
│       └── ci.yml           # Integración continua: instala y corre los tests
│
├── notebooks/
│   ├── 01_exploracion_datos.ipynb
│   └── 02_preprocesamiento.ipynb
│
├── src/
│   ├── download_data.py     # Descarga el dataset de terremotos (espejo en Google Drive)
│   ├── download_pb2002.py   # Descarga los límites de placa PB2002 (espejo en Google Drive)
│   ├── zona_sismica.py      # Calcula el peso de zona sísmica (feature geográfica)
│   └── predict.py           # Carga el modelo y realiza predicciones
│
├── pipelines/
│   ├── clasificacion_fuerte.py  # Entrena el modelo oficial y lo guarda en models/
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
│   ├── sample_database.csv      # Muestra de 2000 filas versionada en Git (tests y fallback)
│   ├── database.csv.dvc         # Puntero DVC del dataset de terremotos
│   └── PB2002_steps.csv.dvc     # Puntero DVC de los límites de placa
│
├── models/
│   └── modelo_clasificador_fuerte.joblib.dvc  # Puntero DVC del modelo
│
└── tests/
    ├── conftest.py          # Rutas e insumos compartidos por los tests
    ├── fixtures/
    │   └── limites_mini.csv # 6 tramos de límite de placa para los tests
    ├── test_data.py         # Carga del dataset y partición
    ├── test_features.py     # Pipeline de features
    └── test_predict.py      # Inferencia end-to-end (código y API)
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
| `.github/workflows/` | Integración continua con GitHub Actions. |
| `dataset-card.md` | Ficha de los datasets: fuente, licencia, columnas, filas y filtros. |
| `models/` | Almacena el modelo entrenado y serializado (versionado con DVC). |
| `data/` | Almacena los datos descargados o generados localmente (versionados con DVC), más una muestra versionada en Git. |

### ¿Qué hace cada módulo en `src/` y `pipelines/`?

| Archivo | Responsabilidad |
|---|---|
| `src/download_data.py` | Descarga el dataset de terremotos a `data/database.csv` desde un espejo público en Google Drive (no requiere credenciales). |
| `src/download_pb2002.py` | Descarga los tramos de límites de placa PB2002 a `data/PB2002_steps.csv` desde Google Drive. |
| `src/zona_sismica.py` | Calcula la feature `peso_zona_sismica` a partir de la ubicación y los límites de placa. |
| `src/predict.py` | Carga el modelo entrenado y realiza predicciones sobre nuevos eventos. |
| `pipelines/preprocesamiento.py` | Carga de datos (con fallback a la muestra), partición 70/20/10 y pipeline de preprocesamiento. |
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
| `make install` | Solo sincroniza las dependencias (`uv sync --locked`). |
| `make kernel` | Registra el entorno como kernel de Jupyter. |
| `make data` | Descarga `database.csv` y `PB2002_steps.csv` desde Google Drive (sin credenciales). |
| `make pull-data` | Trae los datos versionados con DVC (requiere acceso al remoto DVC del equipo). |
| `make train` | Entrena el clasificador con los datos de `data/` y guarda el modelo en `models/`. |
| `make release` | Versiona el modelo con DVC y lo registra con GTO (uso del equipo). |
| `make test` | Evalúa el modelo guardado sobre el conjunto de prueba, sin reentrenar. |
| `make run` | Ejecuta `main.py` dentro del entorno. |
| `make api` | Levanta la API de FastAPI con `uvicorn` y sirve la página web en http://127.0.0.1:8000. |
| `make clean` | Elimina el entorno virtual y las cachés. |
| `make help` | Lista todos los comandos disponibles. |

### 3. Preparar los datos

```bash
make data
```

Descarga los dos datasets a `data/` desde un espejo público en Google Drive, usando `gdown`. No pide credenciales de Kaggle ni de ningún otro servicio. Si los archivos ya existen, no los vuelve a descargar.

Los datos crudos no se guardan en GitHub (están en `.gitignore`). La fuente original del dataset es Kaggle (ver [`dataset-card.md`](dataset-card.md)).

Si `data/database.csv` no está, el código usa automáticamente la muestra versionada `data/sample_database.csv` (2000 filas). Esa muestra existe para los tests y para que el código no falle en un clon sin datos; el modelo oficial se entrena con el dataset completo.

### 4. Entrenar el modelo

```bash
make train
```

Ejecuta `pipelines/clasificacion_fuerte.py`: partición 70/20/10, búsqueda de hiperparámetros con validación cruzada de 5 folds sobre entrenamiento, evaluación en validación y prueba, y guardado del pipeline completo en `models/modelo_clasificador_fuerte.joblib`. Con el dataset completo tarda unos minutos.

`make train` no usa DVC ni GTO. Para versionar y registrar el modelo entrenado, el equipo usa `make release` (ver [Versionado de datos y modelo](#versionado-de-datos-y-modelo-dvc--gto)).

### 4b. Evaluar el modelo guardado

```bash
make test
```

Carga el modelo de `models/` y muestra sus métricas sobre el conjunto de prueba, sin reentrenar. Las pruebas automatizadas se corren con `uv run pytest` (ver [Pruebas automatizadas](#pruebas-automatizadas)).

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

# 2. Descargar los datasets desde Google Drive a data/ (sin credenciales)
make data

# 3. Entrenar el modelo y guardarlo en models/
make train

# 4. Levantar la API y servir la página web en http://127.0.0.1:8000
make api
```

En una sola línea (cada paso solo continúa si el anterior termina bien):

```bash
make setup && make data && make train && make api
```

Al terminar, abre http://127.0.0.1:8000/ en el navegador para usar la interfaz. `make api` queda corriendo en la terminal; se detiene con `Ctrl+C`.

> **Nota:** ningún paso pide credenciales. `make data` descarga de Google Drive y `make train` no depende de DVC. Si quieres verificar el proyecto sin entrenar, corre `uv run pytest`.

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

El versionado está separado del entrenamiento. Después de `make train`, el equipo corre `make release`, que hace `dvc add` y `dvc push` del modelo y registra una nueva versión con **GTO** (`gto register clasificador-sismo-fuerte`). Las versiones registradas hasta ahora están como tags de Git (`clasificador-sismo-fuerte@v1.0.0`, `clasificador-sismo-fuerte@v1.1.0`).

`make release` y `make pull-data` necesitan acceso al remoto DVC del equipo (hoy, una carpeta local). Por eso no forman parte del flujo para un clon nuevo: ese flujo usa `make data` y `make train`.

Para inspeccionar el registro:

```bash
# Listar los artefactos registrados
uv run dvc artifacts list .

# Ver el historial de versiones registradas con GTO
uv run gto history
```

---

## Pruebas automatizadas

```bash
uv run pytest -v
```

Los tests no descargan nada. Usan la muestra versionada `data/sample_database.csv` y un archivo pequeño de límites de placa (`tests/fixtures/limites_mini.csv`, 6 tramos), así que corren igual en local y en CI, sin datos completos ni modelo entrenado. Tardan unos segundos.

| Archivo | Qué verifica |
|---|---|
| `tests/test_data.py` | La muestra carga con 2000 filas y sin nulos en las columnas del modelo; los rangos de latitud, longitud, profundidad y magnitud son válidos; la carga cae en la muestra si falta `database.csv`; la partición 70/20/10 no mezcla filas entre conjuntos. |
| `tests/test_features.py` | `peso_zona_sismica` queda entre 1 y 100 al ajustar y al transformar; el pipeline produce las 4 columnas del modelo con valores finitos; las entradas con nulos se imputan. |
| `tests/test_predict.py` | Inferencia end-to-end: entrena un Random Forest chico (20 árboles), lo serializa con `joblib`, y verifica que `predecir()` y `POST /predict` devuelven una probabilidad entre 0 y 1, que `/health` responde y que una latitud fuera de rango da error 422. |

---

## Integración continua (CI)

El workflow `.github/workflows/ci.yml` corre en GitHub Actions en cada `push` y cada Pull Request:

1. Prepara una máquina Ubuntu limpia y descarga el repositorio.
2. Instala `uv`.
3. Instala Python y las dependencias exactas de `uv.lock` (`uv sync --locked`).
4. Ejecuta `uv run pytest -v`.

Si todos los tests pasan, el commit queda con check verde en GitHub; si alguno falla, o si `uv.lock` no coincide con `pyproject.toml`, queda en rojo. El CI no entrena el modelo oficial ni descarga los datasets completos.

---

## Limitaciones del proyecto

El modelo tendrá las siguientes consideraciones:

- Los datos corresponden a terremotos con magnitud reportada de 5.5 o superior, por lo que no representan la totalidad de los eventos sísmicos.
- La información histórica no garantiza que los patrones identificados se mantengan en futuros eventos.
- La calidad de las predicciones dependerá de la precisión y disponibilidad de los registros originales.
- Un árbol de decisión puede presentar sobreajuste si su complejidad no se controla adecuadamente.
- Las estimaciones del modelo no constituyen predicciones de ocurrencia de terremotos ni permiten determinar cuándo o dónde sucederá uno.
- El entrenamiento no filtra por la columna `Type`, así que incluye 180 eventos que no son terremotos naturales (175 explosiones nucleares, 4 explosiones y 1 *rock burst*) de 23 412.
- La partición es aleatoria, no temporal. El modelo no usa la fecha como variable, pero el desempeño medido puede no reflejar exactamente el de eventos futuros.

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
| gdown | Descarga de los datasets desde Google Drive. |
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
