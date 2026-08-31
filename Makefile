.PHONY: install seed detect train eval kb investigate pipeline all up api web build clean

VENV := backend/.venv

# Installed ahead of requirements.txt so pip resolves the CPU-only wheel. The
# default PyPI wheel drags in the CUDA runtime (~2GB) for a job that never
# touches a GPU. Override to install a CUDA build.
TORCH_INDEX ?= https://download.pytorch.org/whl/cpu

# Stamped so `make seed` doesn't reinstall the world on every run; delete
# backend/.venv (or `make clean`) to force a rebuild.
$(VENV)/.installed: backend/requirements.txt
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install --index-url $(TORCH_INDEX) torch
	$(VENV)/bin/pip install -r backend/requirements.txt
	@touch $@

install: $(VENV)/.installed

seed: install
	@echo "Running the service simulator and anomaly injector..."
	cd backend && .venv/bin/python -m app.seed

detect: install
	@echo "Running naive and Isolation Forest detectors..."
	cd backend && .venv/bin/python -m app.detector_naive
	cd backend && .venv/bin/python -m app.detector_isolation_forest

train: install
	@echo "Training the LSTM autoencoder over multivariate windows..."
	cd backend && .venv/bin/python -m app.detector_lstm_autoencoder

eval: detect train
	@echo "Scoring detectors against ground truth..."
	cd backend && .venv/bin/python -m app.eval_harness

kb: install
	@echo "Building the FAISS index over past incident write-ups..."
	cd backend && .venv/bin/python -m app.knowledge_base

# Investigates 5 incidents spanning all three difficulty tiers. Needs
# ANTHROPIC_API_KEY; append OFFLINE=1 to run the loop against the scripted stub
# in app/offline_agent.py instead (no key, no network, no model).
INVESTIGATE_FLAGS := $(if $(OFFLINE),--offline,)
investigate: install
	cd backend && .venv/bin/python -m app.investigator --sample $(INVESTIGATE_FLAGS)

api: install
	cd backend && .venv/bin/uvicorn app.main:app --reload --port 8000

# One command, empty database to investigated incidents.
# Append OFFLINE=1 to run the agent stage without an API key.
PIPELINE_FLAGS := $(if $(OFFLINE),--offline,)
pipeline: install
	cd backend && .venv/bin/python -m app.pipeline $(PIPELINE_FLAGS) $(ARGS)

# `pipeline` is the same sequence with staged reporting and failure handling.
all: pipeline

up:
	docker compose up --build

# The Vite dev server. Expects `make api` running in another shell.
web:
	cd frontend && npm install && npm run dev

build:
	cd frontend && npm install && npm run lint && npm run build

clean:
	rm -rf $(VENV) backend/telemetry.db backend/artifacts frontend/dist
	find backend -name '__pycache__' -type d -prune -exec rm -rf {} +
