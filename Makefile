install: .venv
	.venv/bin/pip install -r requirements.txt

test:
	.venv/bin/python3 setup.py test

.venv:
	python3 -m venv .venv
