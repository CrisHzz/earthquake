"""Descarga el dataset de terremotos y lo deja en data/database.csv.

Se invoca con: uv run python src/download_data.py (ver `make data`).

El dataset original es "usgs/earthquake-database" de Kaggle (ver README y
dataset-card.md). Para que el proyecto sea reproducible sin credenciales de
Kaggle, se descarga desde un espejo público en Google Drive usando `gdown`.

Es idempotente: si data/database.csv ya existe, no vuelve a descargar.
"""

import sys
from pathlib import Path

import gdown

# ID del archivo en Google Drive (espejo público del dataset de Kaggle).
DRIVE_FILE_ID = "11IPQhBfy0osjn9xt8QW4NziB65O5KHbg"
MAIN_FILE = "database.csv"


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    target = data_dir / MAIN_FILE

    if target.exists():
        print(f"El dataset ya existe en {target}. No se vuelve a descargar.")
        return 0

    data_dir.mkdir(parents=True, exist_ok=True)

    url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
    print(f"Descargando el dataset desde Google Drive ({MAIN_FILE})...")
    try:
        resultado = gdown.download(url, str(target), quiet=False)
    except Exception as exc:  # noqa: BLE001 - mensaje util para el usuario
        print(f"No se pudo descargar el dataset desde Google Drive: {exc}", file=sys.stderr)
        return 1

    if resultado is None or not target.exists():
        print(
            "La descarga no produjo el archivo esperado. Verifica que el enlace de "
            "Google Drive sea publico ('Cualquier persona con el enlace').",
            file=sys.stderr,
        )
        return 1

    print(f"Dataset listo en {target}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
