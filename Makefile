.PHONY: install seed up

install:
	pip install -r backend/requirements.txt

seed:
	@echo "Running the service simulator and anomaly injector..."
	cd backend && python app/seed.py

up:
	docker-compose up --build