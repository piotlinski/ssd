help: ## Show this help
	@grep -E '^[.a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

format: ## Run pre-commit hooks to format code
	 pre-commit run --all-files

args ?= -vvv --cov ssd
test: ## Run tests
	pytest $(args)

shell: ## Run poetry shell
	poetry shell

build: ## Build docker image
	bash -c 'read -sp "PyPI trasee_rd password: " && docker build --build-arg PYPI_PASSWORD="$$REPLY" -f Dockerfile -t ssd:latest .'

docker_args ?= --gpus all  --volume $(pwd):/app --volume $(pwd)/data:/app/data --volume $(pwd)/models:/app/models
train: ## Train model
	docker run  --rm $(docker_args) ssd:latest train

evaluate: ## Evaluate model
	docker run  --rm $(docker_args) ssd:latest evaluate
