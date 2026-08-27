JULIA_PROJECT := src/impute/OneSidedMC

# The analysis runs on the TEXT-ONLY view. The multimodal-inclusive corpus is
# retained and reachable by overriding both roots on the command line, e.g.
#   make preproc DATA_ROOT=data
#   make runall  DATA_ROOT=data RESULTS_ROOT=results
DATA_ROOT    ?= data/text_only
RESULTS_ROOT ?= results/text_only
ROOTS        := --data-root $(DATA_ROOT) --results-root $(RESULTS_ROOT)

# --- parallel runs ------------------------------------------------------------
# Methods are passed as space-separated arguments; each method gets its own
# Rscript process, in parallel (capped at JOBS concurrent):
#
#   make impute softimpute missforest knn
#   make factor softimpute onesidedmc zeros default
#   make loco   softimpute knn              # factor with --loco
#
# Aggregates run the whole default set:
#
#   make runall-impute     all imputations (plain + --raw)
#   make runall-factor     all methods (plain + --raw) + default + zeros
#   make runall-loco       leave-one-column-out factor runs (plain + --raw)
#   make runall            everything above, then the loadings comparison
#
# Every job logs to $(LOGS)/<job>.log; one "[ok] <job>" / "[error] <job>" line
# per job goes to $(LOGS)/summary.txt, with the tail of failing logs inlined.
# ------------------------------------------------------------------------------
JOBS ?= 8
TAIL ?= 50
export JOBS TAIL
LOGS := $(RESULTS_ROOT)/logs

IMPUTE_METHODS := softimpute onesidedmc missforest knn
FACTOR_METHODS := softimpute onesidedmc missforest knn
LOCO_PLAIN     := raw softimpute onesidedmc knn   # plain missforest loco skipped

CLEAR_SUMMARY := @mkdir -p $(LOGS) && : > $(LOGS)/summary.txt

# The method words typed after the target (e.g. "make factor knn missforest").
EXTRA := $(filter-out impute factor loco,$(MAKECMDGOALS))

.PHONY: deps env env-py env-r env-jl preproc clean \
        impute factor loco \
        runall runall-impute runall-factor runall-loco

SUDO := $(shell if [ "$$(id -u)" -eq 0 ]; then echo ""; else echo "sudo"; fi)

deps:
	$(SUDO) apt update -y
	$(SUDO) apt install r-base -y
	curl -fsSL https://install.julialang.org | sh -s -- -y
	curl -LsSf https://astral.sh/uv/install.sh | sh

env: env-py env-r env-jl
	@echo "All environments installed."

export PATH := $(HOME)/.juliaup/bin:$(HOME)/.local/bin:$(PATH)

env-py:
	uv sync

env-r:
	Rscript install.R

env-jl:
	julia --project=$(JULIA_PROJECT) -e 'using Pkg; Pkg.instantiate()'

preproc:
	uv run python scripts/collapse_results.py --data-root $(DATA_ROOT)
	uv run python scripts/densify.py --data-root $(DATA_ROOT)

# --- ad hoc jobs ---------------------------------------------------------------
impute:
	make preproc
	$(CLEAR_SUMMARY)
	./scripts/runmulti.sh $(LOGS) impute -- $(EXTRA) -- $(ROOTS)

factor:
	make preproc
	$(CLEAR_SUMMARY)
	./scripts/runmulti.sh $(LOGS) factor -- $(EXTRA) -- $(ROOTS)

loco:
	make preproc
	$(CLEAR_SUMMARY)
	./scripts/runmulti.sh $(LOGS) factor loco -- $(EXTRA) -- $(ROOTS)

# --- aggregates ---------------------------------------------------------------
runall-impute:
	$(CLEAR_SUMMARY)
	./scripts/runmulti.sh $(LOGS) impute -- $(IMPUTE_METHODS) -- $(ROOTS)
	./scripts/runmulti.sh $(LOGS) impute raw -- $(IMPUTE_METHODS) -- $(ROOTS)

runall-factor:
	$(CLEAR_SUMMARY)
	./scripts/runmulti.sh $(LOGS) factor -- $(FACTOR_METHODS) default zeros -- $(ROOTS)
	./scripts/runmulti.sh $(LOGS) factor raw -- $(FACTOR_METHODS) -- $(ROOTS)

runall-loco:
	$(CLEAR_SUMMARY)
	./scripts/runmulti.sh $(LOGS) factor loco -- $(LOCO_PLAIN) -- $(ROOTS)
	./scripts/runmulti.sh $(LOGS) factor raw loco -- $(FACTOR_METHODS) -- $(ROOTS)

runall:
	$(CLEAR_SUMMARY)
	$(MAKE) --no-print-directory -j$(JOBS) runall-impute runall-factor runall-loco
	./scripts/runone.sh compare-loadings $(LOGS) uv run python scripts/compare_loadings.py --results $(RESULTS_ROOT)

clean:
	cd results && rm -rf *
