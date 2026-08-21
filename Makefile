.PHONY: help install lint test test-fast parity up down replay train smoke bench clean

PY ?= python3
DSN ?= postgresql://fraud:fraud@localhost:5432/fraud

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "};{printf "\033[36m%-16s\033[0m %s\n",$$1,$$2}'

install:  ## install the package with dev, train and spark extras
	$(PY) -m pip install -e ".[dev,train,spark]"

lint:  ## ruff + mypy
	$(PY) -m ruff check src tests
	$(PY) -m ruff format --check src tests
	$(PY) -m mypy

test:  ## full test suite including the local[2] Spark job test
	$(PY) -m pytest

test-fast:  ## everything except Spark (seconds, not a minute)
	$(PY) -m pytest --ignore=tests/test_spark_job.py

parity:  ## just the offline/online byte-identity assertion
	$(PY) -m pytest tests/test_parity_offline_online.py -v

up:  ## start the whole stack
	docker compose up -d --build

down:  ## stop and remove volumes
	docker compose down -v

replay:  ## replay the dataset into Kafka at 100x
	$(PY) -m fraudpipe.replayer --dataset $(DATASET) --path $(DATA) --speedup 100

train:  ## offline replay -> features -> LightGBM -> ONNX
	$(PY) -m fraudpipe.training.build_dataset --dataset $(DATASET) --path $(DATA)
	$(PY) -m fraudpipe.training.train

smoke:  ## publish 100 events end to end, assert 100 decisions land in Postgres
	$(PY) scripts/smoke_compose.py --events 100 --dsn "$(DSN)"

bench:  ## measure sustained throughput and end-to-end latency percentiles
	$(PY) scripts/bench.py --events 20000 --dsn "$(DSN)"

clean:
	rm -rf artifacts .pytest_cache .mypy_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
