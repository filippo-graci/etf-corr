.PHONY: setup test lint format typecheck check tables figures all lock

# Every `uv run` uses the environment as installed by `make setup` (the pinned versions in
# requirements.lock) instead of re-resolving the dependencies.
export UV_NO_SYNC := 1

setup:
	uv venv --python 3.12
	uv pip sync requirements.lock
	uv pip install --no-deps -e .

test:
	uv run pytest -q

lint:
	uv run ruff check . && uv run ruff format --check .

format:
	uv run ruff check --fix . && uv run ruff format .

typecheck:
	uv run mypy

check: lint typecheck test

# Rebuild every table and figure from the data on disk (the data download aside; see the README).
# Runs guarded by a single-run marker (`etfcorr eval/econ --final`, H4, the 2007-2013 replication, E9,
# E10) are never repeated: their recorded outputs in reports/tables stand. H4's validation rehearsal
# needs the separate torch environment and is not included. Expect one to two hours.
tables:
	uv run python scripts/estimator_stability_3_5.py --n-bootstrap 500
	uv run python scripts/mst_figure_3_6.py
	uv run python scripts/h2_table_4_6.py
	uv run python scripts/baseline_table_5_6.py
	uv run python scripts/variants_6_0.py
	uv run python scripts/dm_calibration_6_2.py
	uv run python scripts/gbm_vs_baselines_6_4.py
	uv run etfcorr eval
	uv run etfcorr econ
	uv run python scripts/null_benchmark_8_4.py
	uv run python scripts/followup_h1.py
	uv run python scripts/followup_h3.py
	uv run python scripts/followup_h5.py
	uv run python scripts/followup_h6.py
	uv run python scripts/scoreboard_8_6.py

figures:
	uv run python scripts/figures_8_7.py
	uv run python scripts/figures_followup.py
	uv run python scripts/figures_explained.py

all: tables figures

# requirements.lock holds the versions of every package behind the results. This target re-resolves
# them to the newest allowed versions: the results were not produced with those.
lock:
	uv pip compile pyproject.toml --extra dev --extra xcheck --python-version 3.12 \
		--generate-hashes --no-header -o requirements.lock
