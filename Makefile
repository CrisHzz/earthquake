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

.DEFAULT_GOAL := help

.PHONY: help setup setup-linux install-uv install kernel data run api lab clean

help: ## Muestra esta ayuda
	@echo Comandos disponibles:
	@echo   make setup       - Instala dependencias y registra el kernel (uv ya instalado)
	@echo   make setup-linux - Instala uv si falta + setup completo (Debian/Ubuntu)
	@echo   make install     - Sincroniza dependencias con uv.lock (uv sync --locked)
	@echo   make kernel      - Registra el entorno como kernel de Jupyter
	@echo   make data        - Descarga el dataset de Kaggle a data/database.csv
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

run: ## Ejecuta main.py dentro del entorno
	uv run python main.py

api: ## Levanta la API FastAPI en modo desarrollo
	uv run uvicorn main:app --reload

clean: ## Borra el entorno virtual y las caches de Python
	rm -rf .venv
	rm -rf .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
