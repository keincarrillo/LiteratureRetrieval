# Entorno de desarrollo de las herramientas de LiteratureRetrieval.
#
# Aísla todas las dependencias de Python en un virtualenv local (tools/.venv)
# para no instalar nada en las librerías del host. El flujo de trabajo es
# sobre el texto del libro (.txt) y su corpus (.csv): filter.py y keywords.py.

# --- Configuración -----------------------------------------------------------

# Intérprete con el que se crea el venv: el más nuevo disponible (>=3.10).
PYTHON ?= $(shell for p in python3.13 python3.12 python3.11 python3.10; do \
	if command -v $$p >/dev/null 2>&1; then command -v $$p; break; fi; \
done)

# Rutas absolutas para poder invocar los scripts desde cualquier directorio.
TOOLS_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
ROOT_DIR  := $(abspath $(TOOLS_DIR)/..)
VENV      := $(TOOLS_DIR)/.venv
VENV_PY   := $(VENV)/bin/python

# Datos: el libro del corpus sobre el que se trabaja (carpeta dentro de corpus/).
# Cada libro define sus metadatos y motivos en su propio book.json.
BOOK       ?= Levedad
CORPUS_DIR ?= $(ROOT_DIR)/corpus/$(BOOK)
CONFIG     ?= $(CORPUS_DIR)/book.json
# Solo se pasa --config si el libro define su book.json; si no, el script usa
# sus valores por defecto (comportamiento original de Kundera).
CONFIG_ARG := $(if $(wildcard $(CONFIG)),--config $(CONFIG),)
FILTER_URL ?= http://127.0.0.1:8080

.PHONY: help install shell filter keywords clean

# --- Ayuda -------------------------------------------------------------------

help:
	@echo "Entorno de desarrollo (venv) para las herramientas de LiteratureRetrieval"
	@echo ""
	@echo "  make install   - crea .venv e instala las dependencias (requirements.txt)"
	@echo "  make shell     - abre una shell con el virtualenv activado"
	@echo "  make filter    - abre el revisor web de fragmentos en $(FILTER_URL)"
	@echo "  make keywords  - recalcula las keywords del CSV del corpus"
	@echo "  make clean     - elimina el virtualenv y las cachés"
	@echo ""
	@echo "  Cada libro se configura en su corpus/<libro>/book.json. Modifica según el libro:"
	@echo "    - txt/csv   : rutas del texto y del corpus"
	@echo "    - book_id, title, author, language, source, url : metadatos"
	@echo "    - meta_documentos : nº objetivo de fragmentos a curar"
	@echo "    - motivos_filter  : regex para puntuar/resaltar fragmentos"
	@echo "    - motivos_keywords: etiqueta -> regex para la columna keywords"
	@echo ""
	@echo "  Ejemplo: make filter BOOK=Odisea   /   make keywords BOOK=Odisea"
	@echo ""
	@echo "  Variables: PYTHON=<intérprete> BOOK=<carpeta> CORPUS_DIR=<dir>"

# --- Entorno virtual ---------------------------------------------------------

install:
	@test -n "$(PYTHON)" || (echo "ERROR: se requiere Python >=3.10 (p. ej. brew install python@3.12)"; exit 1)
	@$(PYTHON) -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' \
		|| (echo "ERROR: $(PYTHON) es < 3.10; indica otro con PYTHON=/ruta/python3.12"; exit 1)
	$(PYTHON) -m venv $(VENV)
	$(VENV_PY) -m pip install --upgrade pip
	$(VENV_PY) -m pip install -r $(TOOLS_DIR)/requirements.txt
	@echo ""
	@echo "Entorno listo en $(VENV)"

# Crea el venv automáticamente la primera vez que se use cualquier comando.
$(VENV_PY):
	@$(MAKE) --no-print-directory install

shell: $(VENV_PY)
	@. $(VENV)/bin/activate && exec $${SHELL:-/bin/sh}

# --- Herramientas ------------------------------------------------------------

filter: $(VENV_PY)
	cd $(CORPUS_DIR) && $(VENV_PY) $(TOOLS_DIR)/filter.py $(CONFIG_ARG)

keywords: $(VENV_PY)
	cd $(CORPUS_DIR) && $(VENV_PY) $(TOOLS_DIR)/keywords.py $(CONFIG_ARG)

# --- Limpieza ----------------------------------------------------------------

clean:
	rm -rf $(VENV)
	find $(TOOLS_DIR) -type d -name __pycache__ -prune -exec rm -rf {} +
	@echo "Virtualenv y cachés eliminados."
