# Makefile - EarthQuake Estimator
# Requiere: uv (https://docs.astral.sh/uv/) y make.
# En Windows, instala make con: choco install make  |  scoop install make
#
# Uso rapido:
#   make setup   -> instala todo y deja el entorno listo
#   make help    -> lista los comandos disponibles

# Nombre del kernel de Jupyter que se registra
KERNEL_NAME := earthquake
KERNEL_DISPLAY := earthquake

# Artefacto del modelo oficial (clasificador de terremoto fuerte)
MODELO := models/modelo_clasificador_fuerte.joblib
ARTEFACTO_GTO := clasificador-sismo-fuerte

.DEFAULT_GOAL := help

.PHONY: help setup setup-linux install-uv install kernel data pull-data train test run api lab clean

help: ## Muestra esta ayuda
	@echo Comandos disponibles:
	@echo   make setup       - Instala dependencias y registra el kernel (uv ya instalado)
	@echo   make setup-linux - Instala uv si falta + setup completo (Debian/Ubuntu)
	@echo   make install     - Sincroniza dependencias con uv.lock (uv sync --locked)
	@echo   make kernel      - Registra el entorno como kernel de Jupyter
	@echo   make data        - Descarga el dataset de Kaggle a data/database.csv
	@echo   make pull-data   - Trae los datos versionados (database.csv y PB2002_steps.csv) con DVC
	@echo   make train       - Entrena el clasificador, guarda el modelo y lo versiona (DVC + GTO)
	@echo   make test        - Evalua el modelo guardado sobre el conjunto de prueba (sin reentrenar)
	@echo   make run         - Ejecuta main.py
	@echo   make api         - Levanta la API de FastAPI (uvicorn)
	@echo   make clean       - Elimina el entorno virtual y caches

setup: install kernel ## Setup completo: dependencias + kernel de Jupyter
	@echo.
	@echo Setup completo. Ya puedes empezar a trabajar.
	@echo En tu notebook selecciona el kernel: $(KERNEL_DISPLAY)

setup-linux: install-uv ## Setup para Debian/Ubuntu: instala uv si falta + setup completo
	@echo Verificando dependencias del sistema...
	@command -v make >/dev/null 2>&1 || (echo "Instalando make y curl..." && sudo apt-get update && sudo apt-get install -y make curl)
	@$(MAKE) setup
	@echo.
	@echo Setup de Linux completo.
	@echo Si 'uv' no se reconoce, abre una nueva terminal o ejecuta: source $$HOME/.local/bin/env

install-uv: ## Instala uv en Debian/Ubuntu si no esta presente
	@if command -v uv >/dev/null 2>&1; then \
		echo "uv ya esta instalado: $$(uv --version)"; \
	else \
		echo "Instalando uv..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
		echo "uv instalado. Puede que necesites reiniciar la terminal."; \
	fi

install: ## Sincroniza las dependencias segun uv.lock
	uv sync --locked

kernel: ## Registra el .venv como kernel de Jupyter
	uv run python -m ipykernel install --user --name $(KERNEL_NAME) --display-name "$(KERNEL_DISPLAY)"

pull-data: ## Trae los datos versionados con DVC (database.csv y PB2002_steps.csv)
	uv run dvc pull data/database.csv.dvc data/PB2002_steps.csv.dvc

train: pull-data ## Entrena el clasificador, guarda el modelo y lo versiona con DVC + GTO
	uv run python pipelines/clasificacion_fuerte.py
	uv run dvc add $(MODELO)
	uv run dvc push
	@echo Registrando el modelo con GTO...
	@REM gto register falla si el commit ya esta registrado (modelo identico); no debe tumbar el target
	-uv run gto register $(ARTEFACTO_GTO)

test: pull-data ## Evalua el modelo guardado sobre el conjunto de prueba sin reentrenar
	-uv run dvc pull $(MODELO).dvc
	uv run python pipelines/evaluar.py

run: ## Ejecuta main.py dentro del entorno
	uv run python main.py

api: ## Levanta la API FastAPI y la pagina en http://127.0.0.1:8000
	uv run uvicorn app.api:app --reload

clean: ## Borra el entorno virtual y las caches de Python
	rm -rf .venv
	rm -rf .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
