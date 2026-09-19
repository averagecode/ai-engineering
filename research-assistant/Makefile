# Convenience wrappers. Everything runs locally and costs nothing.
.DEFAULT_GOAL := help
SHELL := /bin/bash
COMPOSE := docker compose
PY := .venv/bin/python

.PHONY: help
help: ## List available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk -F':.*?## ' '{printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# --- containerised workflow (the supported path) ---------------------------- #

.PHONY: up
up: ## Start Weaviate and Ollama
	$(COMPOSE) up -d weaviate ollama

.PHONY: pull-models
pull-models: up ## Download the Ollama chat + embedding models (one-off, ~2.3 GB)
	$(COMPOSE) run --rm model-puller

.PHONY: build
build: ## Build the assistant and test images
	$(COMPOSE) build assistant tests

.PHONY: ingest
ingest: ## Index every PDF in ./data
	$(COMPOSE) run --rm assistant ingest

.PHONY: ask
ask: ## Ask one question: make ask Q="what do these papers conclude?"
	@test -n "$(Q)" || { echo 'Usage: make ask Q="your question"'; exit 1; }
	$(COMPOSE) run --rm assistant ask "$(Q)"

.PHONY: chat
chat: ## Interactive session
	$(COMPOSE) run --rm assistant chat

.PHONY: status
status: ## Show service health and what is indexed
	$(COMPOSE) run --rm assistant status

.PHONY: reset
reset: ## Drop the Weaviate collection
	$(COMPOSE) run --rm assistant reset --yes

.PHONY: test
test: ## Run unit + integration tests in a container against live services
	$(COMPOSE) run --rm tests pytest -v

.PHONY: down
down: ## Stop the services (keeps volumes)
	$(COMPOSE) down

.PHONY: clean
clean: ## Stop services and delete the Weaviate + model volumes
	$(COMPOSE) down -v

.PHONY: logs
logs: ## Tail service logs
	$(COMPOSE) logs -f weaviate ollama

# --- local virtualenv (for development and fast unit tests) ----------------- #

.PHONY: venv
venv: ## Create .venv and install dev dependencies
	python3 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements-dev.txt
	$(PY) -m pip install --no-deps -e .

.PHONY: test-unit
test-unit: ## Run the offline unit tests in .venv (no services needed)
	$(PY) -m pytest -m "not integration" -q

.PHONY: test-integration
test-integration: ## Run the integration tests in .venv against localhost services
	$(PY) -m pytest -m integration -v

.PHONY: coverage
coverage: ## Unit test coverage report
	$(PY) -m pytest -m "not integration" --cov --cov-report=term-missing
