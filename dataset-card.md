# Dataset Card

Documentación de los datos usados por **Earthquake Estimator**. El proyecto usa dos
conjuntos de datos: el dataset principal de terremotos (objetivo del modelo) y un
dataset auxiliar de límites de placas tectónicas (para derivar una feature geográfica).

Ninguno de los dos se versiona en Git: los archivos crudos están en `.gitignore` y se
obtienen con un script (`src/download_data.py`, `src/download_pb2002.py`) y se versionan
con DVC (`data/*.csv.dvc`).

---

## 1. Dataset principal: terremotos (USGS)

| Campo | Valor |
|---|---|
| **Nombre** | Significant Earthquakes, 1965–2016 (Earthquake Database) |
| **Fuente** | [Kaggle – usgs/earthquake-database](https://www.kaggle.com/datasets/usgs/earthquake-database) |
| **Origen primario** | U.S. Geological Survey (USGS) |
| **Archivo** | `data/database.csv` |
| **Descarga** | `src/download_data.py` (vía `kagglehub`, requiere credenciales de Kaggle) → `make data` |
| **Versionado** | DVC → puntero `data/database.csv.dvc` (md5 `6b72dda0…`, ~2.4 MB) |
| **Licencia** | Dominio público (datos de una agencia del gobierno de EE. UU.; ver la página de Kaggle para los términos del dataset) |
| **Nº de filas** | 23 412 |
| **Nº de columnas** | 21 |
| **Rango temporal** | 02/01/1965 – 30/12/2016 |
| **Granularidad** | Un registro por evento sísmico reportado |

### Variable objetivo

El modelo no predice `Magnitude` directamente: se deriva una etiqueta binaria.

- **`fuerte`** = `1` si `Magnitude >= 6.5`, `0` en caso contrario.
- Clase minoritaria: **2 303** eventos fuertes de 23 412 (≈ **9.8 %**). Dataset desbalanceado.
- Rango de `Magnitude` en los datos: **5.5 – 9.1**.

### Columnas

| Columna | Descripción | Nulos | ¿La usa el modelo? |
|---|---|---:|---|
| `Date` | Fecha del evento | 0 | No |
| `Time` | Hora del evento | 0 | No |
| `Latitude` | Latitud del epicentro (grados) | 0 | **Sí (entrada)** |
| `Longitude` | Longitud del epicentro (grados) | 0 | **Sí (entrada)** |
| `Type` | Tipo de evento (ver filtros abajo) | 0 | No |
| `Depth` | Profundidad (km) | 0 | **Sí (entrada)** |
| `Depth Error` | Error de la profundidad | 18 951 | No |
| `Depth Seismic Stations` | Nº de estaciones para la profundidad | 16 315 | No |
| `Magnitude` | Magnitud reportada | 0 | **Sí (deriva la etiqueta)** |
| `Magnitude Type` | Escala de magnitud (MW, MB, …) | 3 | No |
| `Magnitude Error` | Error de la magnitud | 23 085 | No |
| `Magnitude Seismic Stations` | Nº de estaciones para la magnitud | 20 848 | No |
| `Azimuthal Gap` | Separación azimutal entre estaciones | 16 113 | No |
| `Horizontal Distance` | Distancia horizontal de la localización | 21 808 | No |
| `Horizontal Error` | Error de la ubicación horizontal | 22 256 | No |
| `Root Mean Square` | Error RMS del ajuste de señales | 6 060 | No |
| `ID` | Identificador del registro | 0 | No |
| `Source` | Fuente del registro | 0 | No |
| `Location Source` | Fuente de la ubicación | 0 | No |
| `Magnitude Source` | Fuente de la magnitud | 0 | No |
| `Status` | Estado del registro (`Automatic`/`Reviewed`) | 0 | No |

> Las cuatro columnas que el modelo consume (`Latitude`, `Longitude`, `Depth`, `Magnitude`)
> no tienen valores nulos. Varias columnas de error/metadatos tienen muchos nulos
> (hasta ~99 % en `Magnitude Error`), por eso no se usan como features.

### Feature derivada

Además de `Latitude`, `Longitude` y `Depth`, el pipeline calcula internamente
**`peso_zona_sismica`** (escala 1–100), que combina:

- **Componente histórico (60 %)**: actividad sísmica y proporción de eventos fuertes
  por casilla de una cuadrícula lat/lon, aprendido **solo con los datos de
  entrenamiento** (fuera de muestra vía K-fold para evitar fuga).
- **Componente tectónico (40 %)**: distancia al límite de placa más cercano
  (dataset PB2002, ver abajo), ponderada por el tipo de límite.

### Filas filtradas y por qué

- **Partición de datos:** 70 % entrenamiento / 20 % prueba / 10 % validación
  (`train_test_split`, `random_state=42`). El split se hace **antes** de cualquier
  ajuste de preprocesamiento; las estadísticas (imputación, zona sísmica) se ajustan
  solo con entrenamiento para evitar fuga de datos.
- **Nulos:** no se eliminan filas por nulos. Las entradas del modelo no tienen nulos,
  pero el pipeline incluye `SimpleImputer(strategy="median")` como salvaguarda para
  nuevos datos en producción.
- **Tipo de evento (`Type`):** el dataset incluye eventos que no son terremotos
  naturales. Distribución: `Earthquake` 23 232, `Nuclear Explosion` 175,
  `Explosion` 4, `Rock Burst` 1. El pipeline actual **no** filtra por `Type`
  (entrena sobre todas las filas); es una limitación conocida: idealmente se
  conservarían solo los 23 232 eventos `Earthquake`.

---

## 2. Dataset auxiliar: límites de placas PB2002

| Campo | Valor |
|---|---|
| **Nombre** | PB2002 plate boundary steps (Bird, 2003) |
| **Fuente** | [github.com/fraxen/tectonicplates](https://github.com/fraxen/tectonicplates) — `GeoJSON/PB2002_steps.json` |
| **Origen primario** | Bird, P. (2003), *An updated digital model of plate boundaries*, Geochem. Geophys. Geosyst. |
| **Archivo** | `data/PB2002_steps.csv` |
| **Descarga** | `src/download_pb2002.py` → `make data` |
| **Versionado** | DVC → puntero `data/PB2002_steps.csv.dvc` (md5 `f2678aab…`, ~238 KB) |
| **Nº de filas** | 5 824 tramos de límite |
| **Nº de columnas** | 6 |

### Columnas

| Columna | Descripción | Nulos |
|---|---|---:|
| `PLATEBOUND` | Par de placas que forman el límite (p. ej. `AF-AN`) | 0 |
| `STEPCLASS` | Tipo de tramo de límite (ver abajo) | 0 |
| `STARTLONG` | Longitud inicial del tramo | 0 |
| `STARTLAT` | Latitud inicial del tramo | 0 |
| `FINALLONG` | Longitud final del tramo | 0 |
| `FINALLAT` | Latitud final del tramo | 0 |

### Filas / columnas filtradas y por qué

- **Columnas:** del GeoJSON original (~10 MB) se conservaron solo las 6 columnas de
  arriba, reduciendo el archivo a ~238 KB. Son las únicas que necesita
  `src/zona_sismica.py` para calcular distancias a los límites.
- **`STEPCLASS`** se agrupa por tipo de límite para ponderar el riesgo:
  - **Convergente** (peso 1.0): `SUB` (1 129), `OCB` (341), `CCB` (401)
  - **Transformante** (peso 0.6): `OTF` (1 147), `CTF` (457)
  - **Divergente** (peso 0.3): `OSR` (1 875), `CRB` (474)
- No se eliminan filas; se usan los 5 824 tramos.

---

## Reproducir la obtención de los datos

```bash
# Descargar ambos datasets a data/ (requiere credenciales de Kaggle para database.csv)
make data

# O traer las versiones ya registradas con DVC (sin reentrenar ni descargar de Kaggle)
make pull-data
```

Los números de filas, columnas y nulos de esta tarjeta se calcularon sobre
`data/database.csv` (23 412 filas) y `data/PB2002_steps.csv` (5 824 filas)
en la versión versionada con DVC.
