.PHONY: run
run:
	uv run streamlit run main.py --server.port=8502

.PHONY: docker-build
docker-build:
	docker-compose build

.PHONY: docker-up
docker-up:
	docker-compose up

.PHONY: docker-up-build
docker-up-build:
	docker-compose up --build

.PHONY: docker-down
docker-down:
	docker-compose down

.PHONY: docker-logs
docker-logs:
	docker-compose logs -f