flake:
	flake8 katas tests

isort:
	isort katas tests

black:
	black katas tests

mypy:
	mypy katas tests

lint:
	make isort
	make black
	make flake
	make mypy

test:
	pytest tests -vvv