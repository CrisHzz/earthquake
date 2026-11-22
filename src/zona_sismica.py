"""Transformador de scikit-learn que asigna un peso de zona sísmica (1-100).

El peso combina dos componentes calculados para cada par (Latitude, Longitude):

1. **Histórico**: actividad sísmica por casilla de una cuadrícula lat/lon,
   aprendida solo en ``fit()`` con los datos de entrenamiento. Usa el número
   de eventos y la proporción de terremotos fuertes (``Magnitude >= 6.5``),
   suavizada hacia la proporción global y hacia las casillas vecinas.
2. **Tectónico**: distancia al límite de placa más cercano según el modelo
   PB2002 (Bird, 2002), ponderada por el tipo de límite
   (convergente/subducción > transformante > divergente).

Fuente del archivo ``data/PB2002_steps.csv``: ``GeoJSON/PB2002_steps.json`` del
repositorio https://github.com/fraxen/tectonicplates (Bird, P. 2002, *An updated
digital model of plate boundaries*). Se conservaron solo las columnas
``PLATEBOUND``, ``STEPCLASS``, ``STARTLONG``, ``STARTLAT``, ``FINALLONG`` y
``FINALLAT`` de cada tramo para reducir el archivo de 10 MB a 240 KB.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
from sklearn.neighbors import BallTree

RADIO_TIERRA_KM = 6371.0
RUTA_LIMITES = Path(__file__).resolve().parents[1] / "data" / "PB2002_steps.csv"

# Clases de tramo de PB2002 agrupadas por tipo de límite.
GRUPOS_LIMITE = {
    "convergente": ("SUB", "OCB", "CCB"),
    "transformante": ("OTF", "CTF"),
    "divergente": ("OSR", "CRB"),
}
PESOS_LIMITE = {"convergente": 1.0, "transformante": 0.6, "divergente": 0.3}


def _suavizar_vecinos(grid, peso_vecino):
    """Suma a cada casilla sus 8 vecinas multiplicadas por ``peso_vecino``.

    La longitud es circular (la casilla de -180° es vecina de la de 180°);
    la latitud no lo es, por eso las filas de los polos solo tienen vecinas
    hacia el ecuador.
    """
    salida = grid.astype(float).copy()
    for d_fila in (-1, 0, 1):
        for d_col in (-1, 0, 1):
            if d_fila == 0 and d_col == 0:
                continue
            desplazado = np.roll(grid, d_col, axis=1)
            vecino = np.zeros_like(salida)
            if d_fila == 1:
                vecino[1:] = desplazado[:-1]
            elif d_fila == -1:
                vecino[:-1] = desplazado[1:]
            else:
                vecino = desplazado
            salida += peso_vecino * vecino
    return salida


def _densificar_tramos(tabla, paso_deg=0.1):
    """Convierte cada tramo (inicio, fin) en puntos separados ~``paso_deg``."""
    lat0 = tabla["STARTLAT"].to_numpy()
    lon0 = tabla["STARTLONG"].to_numpy()
    lat1 = tabla["FINALLAT"].to_numpy()
    lon1 = tabla["FINALLONG"].to_numpy()
    # Los tramos que cruzan el antimeridiano se interpolan por el lado corto.
    d_lon = (lon1 - lon0 + 180.0) % 360.0 - 180.0
    d_lat = lat1 - lat0
    escala = np.cos(np.radians((lat0 + lat1) / 2.0))
    n_puntos = np.ceil(np.maximum(np.abs(d_lat), np.abs(d_lon) * escala) / paso_deg)
    n_puntos = n_puntos.astype(int) + 1
    lats, lons = [], []
    for i, n in enumerate(n_puntos):
        t = np.linspace(0.0, 1.0, n + 1)
        lats.append(lat0[i] + t * d_lat[i])
        lons.append(lon0[i] + t * d_lon[i])
    lats = np.concatenate(lats)
    lons = (np.concatenate(lons) + 180.0) % 360.0 - 180.0
    return lats, lons, np.repeat(np.arange(len(tabla)), n_puntos + 1)


class PesoZonaSismica(BaseEstimator, TransformerMixin):
    """Agrega la columna ``feature_name`` con un peso sísmico de 1 a 100.

    Parámetros
    ----------
    cell_size_deg : float
        Tamaño de la casilla en grados (2° ≈ 222 km).
    strong_threshold : float
        Magnitud a partir de la cual un terremoto se considera fuerte.
    prior_strength : float
        Número de eventos "virtuales" con la proporción global de terremotos
        fuertes que se suman a cada casilla. Evita valores extremos en
        casillas con pocos eventos.
    neighbor_weight : float
        Peso de cada una de las 8 casillas vecinas al suavizar los conteos.
    decay_km : float
        Distancia a la que la influencia de un límite de placa cae a 1/e.
    hist_weight : float
        Peso del componente histórico; el tectónico recibe ``1 - hist_weight``.
    boundary_weights : dict o None
        Peso por tipo de límite (``convergente``, ``transformante``,
        ``divergente``). ``None`` usa ``PESOS_LIMITE``.
    boundaries_path : str, Path o None
        CSV de tramos PB2002. ``None`` usa ``data/PB2002_steps.csv``.
    n_splits : int
        Particiones para calcular el componente histórico de entrenamiento
        fuera de muestra en ``fit_transform`` (cada fila se evalúa con
        estadísticas que no incluyen su propia magnitud).
    random_state : int
        Semilla de las particiones de ``n_splits``.
    """

    def __init__(
        self,
        cell_size_deg=2.0,
        strong_threshold=6.5,
        prior_strength=20.0,
        neighbor_weight=0.5,
        decay_km=300.0,
        hist_weight=0.6,
        boundary_weights=None,
        boundaries_path=None,
        n_splits=5,
        random_state=42,
        lat_col="Latitude",
        lon_col="Longitude",
        feature_name="peso_zona_sismica",
    ):
        self.cell_size_deg = cell_size_deg
        self.strong_threshold = strong_threshold
        self.prior_strength = prior_strength
        self.neighbor_weight = neighbor_weight
        self.decay_km = decay_km
        self.hist_weight = hist_weight
        self.boundary_weights = boundary_weights
        self.boundaries_path = boundaries_path
        self.n_splits = n_splits
        self.random_state = random_state
        self.lat_col = lat_col
        self.lon_col = lon_col
        self.feature_name = feature_name

    # ------------------------------------------------------------------
    # Cuadrícula
    # ------------------------------------------------------------------
    def _forma_grid(self):
        filas = int(np.ceil(180.0 / self.cell_size_deg))
        columnas = int(np.ceil(360.0 / self.cell_size_deg))
        return filas, columnas

    def _casillas(self, lat, lon):
        filas, columnas = self._forma_grid()
        fila = np.floor((lat + 90.0) / self.cell_size_deg).astype(int)
        col = np.floor(((lon + 180.0) % 360.0) / self.cell_size_deg).astype(int)
        return np.clip(fila, 0, filas - 1), np.clip(col, 0, columnas - 1)

    def _grid_historico(self, lat, lon, y):
        """Puntaje histórico [0, 1] de cada casilla a partir de unos datos."""
        filas, columnas = self._forma_grid()
        fila, col = self._casillas(lat, lon)
        conteo = np.zeros((filas, columnas))
        fuertes = np.zeros((filas, columnas))
        np.add.at(conteo, (fila, col), 1.0)
        np.add.at(fuertes, (fila, col), (y >= self.strong_threshold).astype(float))

        conteo_s = _suavizar_vecinos(conteo, self.neighbor_weight)
        fuertes_s = _suavizar_vecinos(fuertes, self.neighbor_weight)
        prior = fuertes.sum() / conteo.sum()

        actividad = np.log1p(conteo_s) / np.log1p(conteo_s.max())
        tasa = (fuertes_s + self.prior_strength * prior) / (conteo_s + self.prior_strength)
        tasa = tasa / tasa.max()
        grid = 0.5 * actividad + 0.5 * tasa
        return grid, conteo, fuertes, prior

    # ------------------------------------------------------------------
    # Componente tectónico
    # ------------------------------------------------------------------
    def _cargar_limites(self):
        ruta = RUTA_LIMITES if self.boundaries_path is None else Path(self.boundaries_path)
        tabla = pd.read_csv(ruta)
        lats, lons, idx = _densificar_tramos(tabla)
        clases = tabla["STEPCLASS"].to_numpy()[idx]
        pesos = PESOS_LIMITE if self.boundary_weights is None else self.boundary_weights
        self.arboles_limite_ = {}
        self.pesos_limite_ = {}
        for grupo, codigos in GRUPOS_LIMITE.items():
            mascara = np.isin(clases, codigos)
            puntos = np.radians(np.column_stack([lats[mascara], lons[mascara]]))
            self.arboles_limite_[grupo] = BallTree(puntos, metric="haversine")
            self.pesos_limite_[grupo] = float(pesos[grupo])

    def _componente_tectonico(self, lat, lon):
        puntos = np.radians(np.column_stack([lat, lon]))
        puntaje = np.zeros(len(lat))
        for grupo, arbol in self.arboles_limite_.items():
            distancia_km = arbol.query(puntos, k=1)[0][:, 0] * RADIO_TIERRA_KM
            puntaje = np.maximum(
                puntaje, self.pesos_limite_[grupo] * np.exp(-distancia_km / self.decay_km)
            )
        return puntaje

    # ------------------------------------------------------------------
    # API de scikit-learn
    # ------------------------------------------------------------------
    def _coordenadas(self, X):
        lat = X[self.lat_col].to_numpy(dtype=float)
        lon = X[self.lon_col].to_numpy(dtype=float)
        return lat, lon

    def _escalar(self, bruto):
        bajo, alto = self.percentiles_
        escalado = 1.0 + 99.0 * (bruto - bajo) / (alto - bajo)
        return np.clip(escalado, 1.0, 100.0)

    def fit(self, X, y):
        if y is None:
            raise ValueError("PesoZonaSismica necesita y (Magnitude) para el componente histórico.")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("PesoZonaSismica espera un DataFrame con columnas de latitud y longitud.")
        y = np.asarray(y, dtype=float)
        lat, lon = self._coordenadas(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]

        # Estadísticas con todo el entrenamiento: se usan en transform().
        self.grid_historico_, self.conteo_, self.fuertes_, self.prior_fuerte_ = (
            self._grid_historico(lat, lon, y)
        )
        self.n_eventos_fit_ = len(y)
        self._cargar_limites()
        tectonico = self._componente_tectonico(lat, lon)

        # Componente histórico de entrenamiento fuera de muestra: cada fila se
        # evalúa con una cuadrícula ajustada sin su propia magnitud.
        historico_oof = np.zeros(len(y))
        kfold = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        for idx_ajuste, idx_eval in kfold.split(lat):
            grid = self._grid_historico(lat[idx_ajuste], lon[idx_ajuste], y[idx_ajuste])[0]
            fila, col = self._casillas(lat[idx_eval], lon[idx_eval])
            historico_oof[idx_eval] = grid[fila, col]

        bruto = self.hist_weight * historico_oof + (1.0 - self.hist_weight) * tectonico
        self.percentiles_ = tuple(np.percentile(bruto, [1, 99]))
        self.peso_entrenamiento_ = self._escalar(bruto)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        salida = X.copy()
        salida[self.feature_name] = self.peso_entrenamiento_
        return salida

    def componentes(self, X):
        """Devuelve los componentes histórico y tectónico (0-1) sin escalar."""
        lat, lon = self._coordenadas(X)
        fila, col = self._casillas(lat, lon)
        return pd.DataFrame(
            {
                "historico": self.grid_historico_[fila, col],
                "tectonico": self._componente_tectonico(lat, lon),
            },
            index=X.index,
        )

    def transform(self, X):
        comp = self.componentes(X)
        bruto = self.hist_weight * comp["historico"] + (1.0 - self.hist_weight) * comp["tectonico"]
        salida = X.copy()
        salida[self.feature_name] = self._escalar(bruto.to_numpy())
        return salida

    def get_feature_names_out(self, input_features=None):
        return np.append(self.feature_names_in_, self.feature_name).astype(object)
