.PHONY: fmt
fmt:
	poetry run black .
	poetry run isort .
