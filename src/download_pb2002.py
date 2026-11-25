"""Descarga los tramos de límites de placa PB2002 y los deja en data/PB2002_steps.csv.

Se invoca con: uv run python src/download_pb2002.py

Fuente: GeoJSON/PB2002_steps.json del repositorio
https://github.com/fraxen/tectonicplates (Bird, P. 2002, *An updated digital
model of plate boundaries*). Se conservan solo las columnas que usa
``src/zona_sismica.py``: PLATEBOUND, STEPCLASS, STARTLONG, STARTLAT, FINALLONG
y FINALLAT.

Es idempotente: si data/PB2002_steps.csv ya existe, no vuelve a descargar.
Usa solo la biblioteca estándar y pandas (ya es dependencia del proyecto).
"""

import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd

URL = (
    "https://raw.githubusercontent.com/fraxen/tectonicplates/master/"
    "GeoJSON/PB2002_steps.json"
)
COLUMNAS = ["PLATEBOUND", "STEPCLASS", "STARTLONG", "STARTLAT", "FINALLONG", "FINALLAT"]
DESTINO = Path(__file__).resolve().parents[1] / "data" / "PB2002_steps.csv"


def descargar_json(url=URL, intentos=3):
    ultimo_error = None
    for _ in range(intentos):
        try:
            with urllib.request.urlopen(url, timeout=60) as respuesta:
                return json.load(respuesta)
        except Exception as error:  # red, DNS, JSON incompleto...
            ultimo_error = error
    raise RuntimeError(f"No se pudo descargar {url}: {ultimo_error}")


def main():
    if DESTINO.exists():
        print(f"{DESTINO} ya existe; no se descarga de nuevo.")
        return 0

    print("Descargando límites de placa PB2002...")
    try:
        datos = descargar_json()
    except RuntimeError as error:
        print(error)
        print(
            "Si es un error de DNS/red, reintenta. También puedes bajar el archivo\n"
            f"desde el navegador ({URL}) y guardarlo en data/PB2002_steps.json,\n"
            "luego volver a correr este script."
        )
        return 1

    filas = [feature["properties"] for feature in datos["features"]]
    tabla = pd.DataFrame(filas)
    faltantes = [c for c in COLUMNAS if c not in tabla.columns]
    if faltantes:
        print(f"Faltan columnas {faltantes}. Columnas disponibles: {list(tabla.columns)}")
        return 1

    DESTINO.parent.mkdir(parents=True, exist_ok=True)
    tabla[COLUMNAS].to_csv(DESTINO, index=False)
    print(f"Guardado {DESTINO} ({len(tabla)} tramos).")
    return 0


if __name__ == "__main__":
    sys.exit(main())