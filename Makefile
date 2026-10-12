# Mol-Metal — single-file developer entry point.
# All targets are PHONY (no files to collide with); .DEFAULT_GOAL := help.
# Usage:
#   source env.sh           # wire up the repo-local .venv CUDA env first
#   make <target>
#
# Convention: PY points at the repo-local canonical .venv interpreter
# (AGENTS.md §5.9) that env.sh already routes `uv run` at. Override on
# the cmdline if needed:
#   make test PY=python3

PY        ?= /home/hugo/codes/mol-metal/.venv/bin/python
SEED      ?= 0
PROJECT   := $(shell pwd)
PYTEST    := $(PY) -m pytest
SCRIPTS   := $(PROJECT)/molmetal/scripts
TESTS     := $(PROJECT)/molmetal/tests
LOGS      := $(PROJECT)/logs
RPT       := $(PROJECT)/molmetal/reports

.PHONY: help setup eval pitfall test smoke reproduce coverage lint clean paper paper-a-repro paper-a-repro-plan
.DEFAULT_GOAL := help

# ----- help -----------------------------------------------------------------
help:                       ## list all targets with one-line summaries
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / \
		{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# ----- env wiring -----------------------------------------------------------
setup:                      ## activate env.sh + verify .venv (idempotent)
	@echo "Run:  source env.sh   (wires UV_PROJECT_ENVIRONMENT at the repo-local .venv)"
	@test -x "$(PY)" || { \
	    echo "ERROR: $(PY) not found — run 'source env.sh' first."; \
	    exit 1; }

# ----- evaluation -----------------------------------------------------------
eval:                       ## run eval_all_evaluators.py (all metrics)
	cd $(PROJECT) && $(PY) $(SCRIPTS)/eval_all_evaluators.py

pitfall:                    ## run wf_pitfall_guard/check_all.py (CI gates)
	cd $(PROJECT) && $(PY) $(SCRIPTS)/wf_pitfall_guard/check_all.py

# ----- unit tests -----------------------------------------------------------
test:                       ## pytest molmetal/tests/ -q (fast)
	cd $(PROJECT) && $(PYTEST) $(TESTS) -q

coverage:                   ## pytest with --cov (HTML + .coverage)
	cd $(PROJECT) && $(PYTEST) $(TESTS) --cov=molmetal --cov-report=html \
	    --cov-report=term-missing

lint:                       ## optional flake8 on molmetal/ (skip if missing)
	@command -v flake8 >/dev/null 2>&1 || { \
	    echo "flake8 not installed; skipping."; exit 0; }
	cd $(PROJECT) && flake8 molmetal/ --max-line-length=100

# ----- CFM smoke (200-step R17 retrain) -------------------------------------
smoke:                      ## 200-step CFM smoke (metallo_drug_smoke_retrain)
	cd $(PROJECT) && $(PY) $(SCRIPTS)/metallo_drug_smoke_retrain.py \
	    --smoke --seed $(SEED) --hidden-dim 64 --n-layers 2 \
	    --out-dir $(RPT)/wf_makefile_smoke

# ----- R17 full reproduce ---------------------------------------------------
reproduce:                  ## bash molmetal/scripts/wf_reproduce_R17.sh
	cd $(PROJECT) && bash $(SCRIPTS)/wf_reproduce_R17.sh

# ----- Paper-A metric reproducibility ---------------------------------------
paper-a-repro:              ## regenerate + diff every CPU-reproducible Paper-A metric
	cd $(PROJECT) && bash scripts/reproduce_paper_a.sh

paper-a-repro-plan:         ## print the Paper-A repro plan (run nothing)
	cd $(PROJECT) && bash scripts/reproduce_paper_a.sh --dry-run

# ----- paper build (4-pass pdflatex + bibtex) -------------------------------
paper:                      ## cd paper && 4-pass pdflatex+bibtex build
	cd $(PROJECT)/paper && pdflatex -interaction=nonstopmode -halt-on-error main.tex \
	    > /tmp/pdflatex_pass1.log 2>&1
	cd $(PROJECT)/paper && bibtex main > /tmp/bibtex_pass2.log 2>&1
	cd $(PROJECT)/paper && pdflatex -interaction=nonstopmode -halt-on-error main.tex \
	    > /tmp/pdflatex_pass3.log 2>&1
	cd $(PROJECT)/paper && pdflatex -interaction=nonstopmode -halt-on-error main.tex \
	    > /tmp/pdflatex_pass4.log 2>&1
	@echo "paper/main.pdf rebuilt (4 passes)."

# ----- housekeeping ---------------------------------------------------------
clean:                      ## rm logs/ htmlcov/ .coverage (keep .venv + reports)
	rm -rf $(LOGS) $(PROJECT)/htmlcov $(PROJECT)/.coverage \
	    $(PROJECT)/.coverage.* $(PROJECT)/.pytest_cache
	@echo "cleaned logs/, htmlcov/, .coverage*"

# ----- meta -----------------------------------------------------------------
.PHONY: list
list: help                  ## alias for `help`
