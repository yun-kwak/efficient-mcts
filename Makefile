.PHONY: install test smoke forge-smoke paper-dry-run

install:
	python -m pip install -e pine -r requirements/py310-cpu.txt

test:
	python -m pytest -q pine/tests

smoke:
	bash scripts/smoke.sh

forge-smoke:
	bash scripts/submit_forge_smoke.sh

paper-dry-run:
	python scripts/run_paper_preset.py --task doorkey_easy --method ours --seed 1 --dry_run
