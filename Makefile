DOCKER_RUN := docker run --rm -v $(shell pwd):/app
LOCAL_USER := -e LOCAL_USER_ID=`id -u $(USER)` -e LOCAL_GROUP_ID=`id -g $(USER)`
DOCKER_ARGS ?=
tag = piotrekzie100/dev:ssd

help:  ## Show this help
	@grep -E '^[.a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

shell:  ## Run virtualenv shell
	poetry shell

args ?= -vvv --cov pytorch_ssd
test:  ## Run tests
	pytest $(args)

gpu ?= 3
ssd_args ?= ssd --default_root_dir runs
run:  ## Run model
	$(DOCKER_RUN) $(LOCAL_USER) $(DOCKER_ARGS) --gpus '"device=$(gpu)"' --shm-size 24G $(tag) $(ssd_args)

cmd ?= python3 train.py $(ssd_args)
run.basic:  ## Run model using basic docker
	$(DOCKER_RUN) $(LOCAL_USER) $(DOCKER_ARGS) --gpus '"device=$(gpu)"' --shm-size 24G --cpus 16 piotrekzie100/dev:basic $(cmd)
