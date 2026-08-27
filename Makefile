JULIA_PROJECT := src/impute/OneSidedMC

# The analysis runs on the TEXT-ONLY view. The multimodal-inclusive corpus is
# retained and reachable by overriding both roots on the command line, e.g.
#   make preproc DATA_ROOT=data
#   make runall  DATA_ROOT=data RESULTS_ROOT=results
# These match the scripts' own defaults; they are stated explicitly here so the
# tree a target writes to is visible at the call site.
DATA_ROOT    ?= data/text_only
RESULTS_ROOT ?= results/text_only
ROOTS        := --data-root $(DATA_ROOT) --results-root $(RESULTS_ROOT)

# --- parallel runs ------------------------------------------------------------
# Each runall directive fans its jobs out as independent processes through a
# sub-make capped at JOBS concurrent processes:
#
#   make runall            imputations + factors in parallel, then loadings compare
#   make runall-impute     all imputation methods (plain + --raw)
#   make runall-factor     baselines (default, zeros) + all methods (plain + --raw)
#   make runall-loco       leave-one-column-out factor runs (plain + --raw)
#   make impute-knn-raw    single ad hoc job (same naming scheme as above)
#
# Override parallelism with JOBS=N on any of these.
#
# Instead of each script's own logging, output goes to
#   $(LOGS)/<job>.log        full stdout/stderr of that job
#   $(LOGS)/summary.txt      one "[ok] <job>" / "[error] <job>" line per job;
#                            for failures, the last TAIL log lines follow inline
# ------------------------------------------------------------------------------
JOBS ?= 4
TAIL ?= 50
export TAIL
LOGS := $(RESULTS_ROOT)/logs

IMPUTE_METHODS := softimpute onesidedmc missforest knn
IMPUTE_TGTS := $(foreach m,$(IMPUTE_METHODS),impute-$m impute-raw-$m)

FACTOR_METHODS := softimpute onesidedmc missforest knn
FACTOR_TGTS := $(foreach m,$(FACTOR_METHODS),factor-$m factor-raw-$m)

# plain --loco missforest is deliberately excluded; raw loco variants all kept.
LOCO_TGTS := $(filter-out factor-loco-missforest,$(foreach m,$(FACTOR_METHODS),factor-loco-$m factor-raw-loco-$m)) factor-loco-raw

CLEAR_SUMMARY := @mkdir -p $(LOGS) && : > $(LOGS)/summary.txt

.PHONY: deps env env-py env-r env-jl preproc clean \
        runall runall-impute runall-factor runall-loco \
        factor-default factor-zeros \
        $(IMPUTE_TGTS) $(FACTOR_TGTS) $(LOCO_TGTS)

SUDO := $(shell if [ "$$(id -u)" -eq 0 ]; then echo ""; else echo "sudo"; fi)

deps:
	$(SUDO) apt update -y
	$(SUDO) apt install r-base -y
	curl -fsSL https://install.julialang.org | sh -s -- -y
	curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all three environments: Python (uv), R (install.R), Julia (OSMC project).
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

# --- aggregates ---------------------------------------------------------------

runall:
	$(CLEAR_SUMMARY)
	-+$(MAKE) --no-print-directory -k -j$(JOBS) $(IMPUTE_TGTS) $(FACTOR_TGTS)
	./scripts/runone.sh compare-loadings $(LOGS) uv run python scripts/compare_loadings.py --results $(RESULTS_ROOT)

runall-impute:
	$(CLEAR_SUMMARY)
	+$(MAKE) --no-print-directory -k -j$(JOBS) $(IMPUTE_TGTS)

runall-factor:
	$(CLEAR_SUMMARY)
	+$(MAKE) --no-print-directory -k -j$(JOBS) factor-default factor-zeros $(FACTOR_TGTS)

runall-loco:
	$(CLEAR_SUMMARY)
	+$(MAKE) --no-print-directory -k -j$(JOBS) $(LOCO_TGTS)

# --- job rules ----------------------------------------------------------------
# Each leaf wraps its command in scripts/runone.sh, which redirects output to
# $(LOGS)/<target>.log and reports [ok]/[error] into summary.txt.
#
# Pattern-rule stems resolve greedily but GNU make prefers the pattern whose
# stem is shortest, so e.g. "factor-raw-loco-knn" binds to factor-raw-loco-%
# (stem knn), not factor-% (stem raw-loco-knn). Methods named like a variant
# flag ("default", "zeros", "loco-raw") must be explicit targets instead.

impute-%:
	./scripts/runone.sh $@ $(LOGS) Rscript src/run/impute.R --method $* --reimpute $(ROOTS)

impute-raw-%:
	./scripts/runone.sh $@ $(LOGS) Rscript src/run/impute.R --method $* --reimpute --raw $(ROOTS)

factor-%:
	./scripts/runone.sh $@ $(LOGS) Rscript src/run/factor.R --method $* $(ROOTS)

factor-raw-%:
	./scripts/runone.sh $@ $(LOGS) Rscript src/run/factor.R --method $* --raw $(ROOTS)

factor-loco-%:
	./scripts/runone.sh $@ $(LOGS) Rscript src/run/factor.R --method $* --loco $(ROOTS)

factor-raw-loco-%:
	./scripts/runone.sh $@ $(LOGS) Rscript src/run/factor.R --method $* --raw --loco $(ROOTS)

factor-default:
	./scripts/runone.sh $@ $(LOGS) Rscript src/run/factor.R --method default $(ROOTS)

factor-zeros:
	./scripts/runone.sh $@ $(LOGS) Rscript src/run/factor.R --method zeros $(ROOTS)

factor-loco-raw:
	./scripts/runone.sh $@ $(LOGS) Rscript src/run/factor.R --method raw --loco $(ROOTS)

clean:
	cd results && rm -rf *
