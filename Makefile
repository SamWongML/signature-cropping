.PHONY: install lint test bench docker run-api run-mcp clean fetch-detr fetch-yolo

PY ?= python3.11
VENV ?= .venv
PIP := $(VENV)/bin/pip
PYBIN := $(VENV)/bin/python

install:
	$(PY) -m venv $(VENV)
	$(PIP) install --upgrade pip wheel
	$(PIP) install -e ".[dev]"

lint:
	$(VENV)/bin/ruff check src tests
	$(VENV)/bin/mypy src

test:
	$(VENV)/bin/pytest -q

bench:
	$(PYBIN) bench/latency.py

fetch-detr:
	$(PYBIN) scripts/fetch_pretrained.py --backend conditional-detr --out-dir models

fetch-yolo:
	$(PYBIN) scripts/fetch_pretrained.py --backend yolov8 --out-dir models

docker:
	docker build --platform=linux/amd64 -t sigcrop:dev .

run-api:
	$(VENV)/bin/uvicorn sigcrop.api.app:app --host 0.0.0.0 --port 8080 --reload

run-mcp:
	$(PYBIN) -m sigcrop.mcp.server

clean:
	rm -rf $(VENV) .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
