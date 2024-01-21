help:  ## Show help
	@grep -E '^[.a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

clean: ## Clean autogenerated files
	rm -rf dist
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	rm -f .coverage

format: ## Run pre-commit hooks
	poetry run pre-commit run -a

sync: ## Merge changes from main branch to your current branch
	git fetch
	git pull

test: ## Run not slow tests
	poetry run pytest -v

test-full: ## Run all tests and coverage.
	poetry run pytest -v --slow

type:
	poetry run mypy .

run: format test-full type

docker-build: ## Build docker image.
	docker build -t ami --no-cache .

docker-run: ## Run built docker image.
	docker run -it --gpus all \
	--mount type=volume,source=ami,target=/workspace \
	ami
