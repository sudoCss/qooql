# Makefile
.PHONY: db services data training search services/data services/training services/search ui run testing

DEFAULT_TARGET: run

db:
	@python -m db.setup

services/data:
	@python -m uvicorn api.data_loader_api:app --host 0.0.0.0 --port 8001

services/training:
	@python -m uvicorn api.representation_api:app --host 0.0.0.0 --port 8002

services/search:
	@python -m uvicorn api.search_api:app --host 0.0.0.0 --port 8003

services/rag:
	@python -m uvicorn api.rag_api:app --host 0.0.0.0 --port 8004

ui:
	@python -m ui.app

ui/web:
	@python -m uvicorn ui_web.app:app --host 0.0.0.0 --port 8000

testing:
	@python -m testing.app

run: ui
