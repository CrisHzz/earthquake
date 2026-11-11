"""Descarga el dataset de terremotos de Kaggle y lo deja en data/database.csv.

Se invoca con: uv run python src/download_data.py (ver `make data`).

Usa kagglehub (dependencia ya declarada) para descargar el dataset
"usgs/earthquake-database". kagglehub lo guarda en su cache; este script
copia el archivo principal database.csv a la carpeta data/ del proyecto,
en la ruta exacta que espera el notebook de exploracion.

Es idempotente: si data/database.csv ya existe, no vuelve a descargar.
"""

import shutil
import sys
from pathlib import Path

DATASET = "usgs/earthquake-database"
MAIN_FILE = "database.csv"

CREDENTIALS_HELP = (
    "No se pudo descargar el dataset desde Kaggle.\n"
    f"El dataset '{DATASET}' requiere credenciales de la API de Kaggle.\n\n"
    "Configura tus credenciales de alguna de estas formas (sin crear archivos "
    "de secretos en el repo):\n"
    "  - Variables de entorno:\n"
    "      KAGGLE_USERNAME=<tu_usuario>\n"
    "      KAGGLE_KEY=<tu_api_key>\n"
    "  - o coloca kaggle.json en %USERPROFILE%\\.kaggle\\kaggle.json (Windows)\n"
    "    o en ~/.kaggle/kaggle.json (Linux/macOS).\n\n"
    "Puedes generar tu API key en https://www.kaggle.com/settings "
    "(seccion 'API' -> 'Create New Token').\n"
    "Tambien debes aceptar los terminos del dataset en:\n"
    "  https://www.kaggle.com/datasets/usgs/earthquake-database"
)


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    target = data_dir / MAIN_FILE

    if target.exists():
        print(f"El dataset ya existe en {target}. No se vuelve a descargar.")
        return 0

    try:
        import kagglehub
    except ImportError:
        print(
            "No se encontro kagglehub. Instala las dependencias con 'make install' "
            "o 'uv sync'.",
            file=sys.stderr,
        )
        return 1

    print(f"Descargando el dataset '{DATASET}' desde Kaggle...")
    try:
        download_path = Path(kagglehub.dataset_download(DATASET))
    except Exception as exc:  # noqa: BLE001 - mensaje util para el usuario
        print(CREDENTIALS_HELP, file=sys.stderr)
        print(f"\nDetalle del error: {exc}", file=sys.stderr)
        return 1

    source = download_path / MAIN_FILE
    if not source.exists():
        # Busca el archivo por si la estructura del archivo descargado cambia.
        candidates = list(download_path.rglob(MAIN_FILE))
        if not candidates:
            print(
                f"No se encontro '{MAIN_FILE}' dentro de la descarga en "
                f"{download_path}.",
                file=sys.stderr,
            )
            return 1
        source = candidates[0]

    data_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    print(f"Dataset listo en {target}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
