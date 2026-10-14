PY := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: install install-research test download extract preflight translate qa package embed rq1 rq2 all

install:
	$(PIP) install -e .[test]

install-research:
	$(PIP) install -e .[research]

test:
	$(PY) -m pytest

download:
	bash scripts/download_data.sh

extract:
	$(PY) -m korean_proverb.extract

preflight:
	$(PY) -m korean_proverb.preflight

translate:
	$(PY) -m korean_proverb.translate

qa:
	$(PY) -m korean_proverb.qa

package:
	$(PY) -m korean_proverb.package

embed:
	$(PY) -m korean_proverb.embed --models $(MODELS)

modelcheck:
	$(PY) -m korean_proverb.model_check --models $(MODELS)

embed-server:
	$(PY) scripts/embed_server.py --host 0.0.0.0 --port 8800

rq2:
	$(PY) -m korean_proverb.rq2

all: download extract preflight translate qa package embed rq1 rq2
