"""Configuración compartida de pytest.

Los tests no descargan nada: usan la muestra versionada
(data/sample_database.csv) y un archivo pequeño de límites de placa
(tests/fixtures/limites_mini.csv), así corren igual en local y en CI.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for ruta in (PROJECT_ROOT, PROJECT_ROOT / "src", PROJECT_ROOT / "pipelines"):
    ruta = str(ruta)
    if ruta not in sys.path:
        sys.path.insert(0, ruta)

RUTA_MUESTRA = PROJECT_ROOT / "data" / "sample_database.csv"
RUTA_LIMITES_MINI = Path(__file__).resolve().parent / "fixtures" / "limites_mini.csv"


@pytest.fixture(scope="session")
def datos_muestra():
    from preprocesamiento import cargar_datos

    return cargar_datos(RUTA_MUESTRA)


@pytest.fixture(scope="session")
def ruta_limites():
    return RUTA_LIMITES_MINI
