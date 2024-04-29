install: .venv
	.venv/bin/pip install -r requirements.txt

test:
	.venv/bin/python3 -m unittest discover tests

.venv:
	python3 -m venv .venv
