help:  ## Show this help
	@grep -E '^[.a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

shell:  ## Run virtualenv shell
	poetry shell

args ?= -vvv --cov pytorch_ssd
test:  ## Run tests
	pytest $(args)

