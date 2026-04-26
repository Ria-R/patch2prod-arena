.PHONY: dev demo docker

dev:
	bash scripts/dev.sh

demo:
	python3 -m patch2prod.cli_demo

docker:
	docker compose up --build
