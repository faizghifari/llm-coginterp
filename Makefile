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
#   make factor-timed [methods]             # year-separated EFA (skips
#                                           # fill-smooth default/zeros)
#
# Aggregates run the whole default set:
#
#   make runall-impute     all imputations (plain + --raw)
#   make runall-factor     all methods (plain + --raw)
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

IMPUTE_METHODS := softimpute onesidedmc missforest knn default zeros
FACTOR_METHODS := softimpute onesidedmc missforest knn default zeros
LOCO_PLAIN     := default zeros softimpute onesidedmc knn   # plain missforest loco skipped

CLEAR_SUMMARY := @mkdir -p $(LOGS) && : > $(LOGS)/summary.txt

# The method words typed after the target (e.g. "make factor knn missforest").
# "factor-timed" is itself a target, so keep it out of the method-word list.
EXTRA := $(filter-out impute factor factor-timed loco,$(MAKECMDGOALS))

.PHONY: deps env env-py env-r env-jl preproc clean \
        impute factor factor-timed loco release-date release-date-report \
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
	$(CLEAR_SUMMARY)
	./scripts/runmulti.sh $(LOGS) impute -- $(EXTRA) -- $(ROOTS)
	./scripts/runmulti.sh $(LOGS) impute raw -- $(EXTRA) -- $(ROOTS)

factor:
	$(CLEAR_SUMMARY)
	./scripts/runmulti.sh $(LOGS) factor -- $(EXTRA) -- $(ROOTS)
	./scripts/runmulti.sh $(LOGS) factor raw -- $(EXTRA) -- $(ROOTS)

loco:
	$(CLEAR_SUMMARY)
	./scripts/runmulti.sh $(LOGS) factor loco -- $(EXTRA) -- $(ROOTS)
	./scripts/runmulti.sh $(LOGS) factor raw loco -- $(EXTRA) -- $(ROOTS)

# Year-separated factoring: one EFA per release-year cohort of models.
# Fill-smooth methods (default/zeros) are excluded by construction — their
# imputed correlation is global, not per cohort — and the R script
# double-guards against them. Method words after the target override
# the default set, e.g. "make factor-timed knn".
factor-timed:
	$(CLEAR_SUMMARY)
	./scripts/runmulti.sh $(LOGS) factor timed -- $(if $(EXTRA),$(EXTRA),$(FACTOR_METHODS)) -- $(ROOTS)
	./scripts/runmulti.sh $(LOGS) factor timed raw -- $(if $(EXTRA),$(EXTRA),$(FACTOR_METHODS)) -- $(ROOTS)

# Release-date analysis (paper appendix "Release-date analysis"). Needs `make
# factor` on the same roots first. Models are binned into release cohorts
# (<=2022, 2023, 2024, >=2025) and each cohort is factored with the row-preserving
# imputers that pass factor.R's R2_GATE, against 50 random subsets of the same
# size. Also: g level per cohort, an imputation-free check on the benchmarks
# every cohort took, and benchmark release year vs the pooled solutions.
# Writes $(RESULTS_ROOT)/release_date/ (summary.md has every table).
#   make release-date DATA_ROOT=<d> RESULTS_ROOT=<r>
release-date:
	uv run python scripts/release_date.py run $(ROOTS)

# Combine release-date runs (one per target density) into the paper's tables
# and figure. RUNS is a space-separated list of LABEL=<results-root>/release_date.
#   make release-date-report RUNS="10%=results/a/release_date 20%=results/b/release_date"
RUNS ?= $(RESULTS_ROOT)/release_date
release-date-report:
	uv run python scripts/release_date.py report $(foreach r,$(RUNS),--run $(r)) \
		--out $(RESULTS_ROOT)/release_date_report

# The method words after the target (e.g. "usvt" in "make factor usvt") arrive as
# goals make wants to build; they're consumed by runmulti via $(EXTRA) above, so
# give them an empty recipe here so make doesn't error "No rule to make target".
.PHONY: $(EXTRA)
$(EXTRA):

# --- aggregates ---------------------------------------------------------------
runall-impute:
	$(CLEAR_SUMMARY)
	./scripts/runmulti.sh $(LOGS) impute -- $(IMPUTE_METHODS) -- $(ROOTS)
	./scripts/runmulti.sh $(LOGS) impute raw -- $(IMPUTE_METHODS) -- $(ROOTS)

runall-factor:
	$(CLEAR_SUMMARY)
	./scripts/runmulti.sh $(LOGS) factor -- $(FACTOR_METHODS) -- $(ROOTS)
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
