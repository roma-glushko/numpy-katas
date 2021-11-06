flake:
	flake8 ./katas

isort:
	isort ./katas

black:
	black ./katas

mypy:
	mypy ./katas

lint:
	make isort
	make black
	make flake
	make mypy


