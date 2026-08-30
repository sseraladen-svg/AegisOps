.PHONY: install seed detect eval up

install:
	python3 -m venv backend/.venv
	backend/.venv/bin/pip install --upgrade pip
	backend/.venv/bin/pip install -r backend/requirements.txt

seed: install
	@echo "Running the service simulator and anomaly injector..."
	cd backend && .venv/bin/python app/seed.py

detect:
	@echo "Running naive and Isolation Forest detectors..."
	cd backend && .venv/bin/python app/detector_naive.py
	cd backend && .venv/bin/python app/detector_isolation_forest.py

eval: detect
	@echo "Scoring detectors against ground truth..."
	cd backend && .venv/bin/python app/eval_harness.py

up:
	docker-compose up --build