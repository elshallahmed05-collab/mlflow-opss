install:
	python -m pip install -r requirements.txt

train:
	python -m src.train

test:
	pytest -q

up:
	docker compose up --build

down:
	docker compose down
