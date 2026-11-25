"""Descarga los tramos de límites de placa PB2002 y los deja en data/PB2002_steps.csv.

Se invoca con: uv run python src/download_pb2002.py (ver `make data`).

Fuente original: GeoJSON/PB2002_steps.json del repositorio
https://github.com/fraxen/tectonicplates (Bird, P. 2003, *An updated digital
model of plate boundaries*). Se conservan solo las columnas que usa
``src/zona_sismica.py``: PLATEBOUND, STEPCLASS, STARTLONG, STARTLAT, FINALLONG
y FINALLAT.

Para que el proyecto sea reproducible sin depender de la estructura del repo
original, se descarga el CSV ya procesado desde un espejo público en Google
Drive usando `gdown`.

Es idempotente: si data/PB2002_steps.csv ya existe, no vuelve a descargar.
"""

import sys
from pathlib import Path

import gdown

# ID del archivo en Google Drive (CSV ya procesado con las 6 columnas).
DRIVE_FILE_ID = "1KlKIyGigCIYIrU_dq2kHmXvGQ_aNqI9n"
DESTINO = Path(__file__).resolve().parents[1] / "data" / "PB2002_steps.csv"


def main() -> int:
    if DESTINO.exists():
        print(f"{DESTINO} ya existe; no se descarga de nuevo.")
        return 0

    DESTINO.parent.mkdir(parents=True, exist_ok=True)

    url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
    print("Descargando límites de placa PB2002 desde Google Drive...")
    try:
        resultado = gdown.download(url, str(DESTINO), quiet=False)
    except Exception as exc:  # noqa: BLE001 - mensaje util para el usuario
        print(f"No se pudo descargar PB2002 desde Google Drive: {exc}", file=sys.stderr)
        return 1

    if resultado is None or not DESTINO.exists():
        print(
            "La descarga no produjo el archivo esperado. Verifica que el enlace de "
            "Google Drive sea publico ('Cualquier persona con el enlace').",
            file=sys.stderr,
        )
        return 1

    print(f"Guardado {DESTINO}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
