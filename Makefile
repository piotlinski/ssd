DOCKER_RUN := docker run -u `id -u $(USER)`:`id -g $(USER)` --rm -v $(shell pwd):/app
tag = piotrekzie100/dev:ssd

help: ## Show this help
	@grep -E '^[.a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

format: ## Run pre-commit hooks to format code
	 pre-commit run --all-files

build.dev: ## Build docker development image
	docker build -f dockerfiles/Dockerfile.dev -t $(tag)-dev .

WANDB_API_KEY ?=
build.prod: ## Build docker production image
	docker build  --build-arg WANDB_API_KEY=$(WANDB_API_KEY) -f dockerfiles/Dockerfile.prod -t $(tag) .

shell: ## Run docker dev shell
	$(DOCKER_RUN) -it $(tag)-dev /bin/bash

args ?= -vvv --cov pyssd
test: ## Run tests
	poetry run pytest $(args)

gpu ?= 3
ssd_args ?= ssd --default_root_dir runs
run: ## Run model
	$(DOCKER_RUN) --gpus '"device=$(gpu)"' --shm-size 24G $(tag) $(ssd_args)
