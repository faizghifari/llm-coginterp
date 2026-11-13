JULIA_PROJECT := src/impute/OneSidedMC

# The analysis runs on the TEXT-ONLY view, at an explicit density. DENSITY is
# required (it names the density slice, e.g. 0.1); both roots gain that segment:
#   data/text_only/<DENSITY>/   results/text_only/<DENSITY>/
# The multimodal-inclusive corpus is retained and reachable by overriding the
# roots on the command line, e.g.
#   make preproc DATA_ROOT=data DENSITY=0.1
#   make runall  DATA_ROOT=data RESULTS_ROOT=results DENSITY=0.1
DATA_ROOT    ?= data/text_only
RESULTS_ROOT ?= results/text_only
# Setup targets have no density dimension (the Docker build runs `make env-*`
# with no DENSITY), so they are exempt from the check. The synth corpus goals
# REQUIRE DENSITY: make_mnar.py and impute.R --synth nest a density segment
# into data/synth/<DENSITY>/ and results/synth/<DENSITY>/.
NO_DENSITY_GOALS := deps env env-py env-r env-jl clean
ifeq ($(filter $(NO_DENSITY_GOALS),$(MAKECMDGOALS)),)
ifeq ($(strip $(DENSITY)),)
$(error DENSITY is required, e.g. make runall DENSITY=0.1)
endif
endif
DENSITY_DATA    := $(DATA_ROOT)/$(DENSITY)
DENSITY_RESULTS := $(RESULTS_ROOT)/$(DENSITY)
ROOTS        := --data-root $(DATA_ROOT) --results-root $(RESULTS_ROOT) --density $(DENSITY)

# --- parallel runs ------------------------------------------------------------
# Methods are passed as space-separated arguments; each method gets its own
# Rscript process, in parallel (capped at JOBS concurrent):
#
#   make impute softimpute missforest knn
#   make factor softimpute onesidedmc zeros default
#   make loco   softimpute knn              # factor with --loco
#   make factor-timed [methods]             # year-separated EFA (skips
#                                           # fill-smooth default/zeros)
#   make impute-synth [methods]             # --synth: all data/synth/<DENSITY>
#                                           # roots (default: missforest softimpute)
#   make impute-synth-hier [methods]        # --hier --synth: density descent
#                                           # per (cell, mask)
#
# The --hier targets are the exception to the method-per-process rule: methods
# run SERIALLY and each method runs its independent dataset chains in parallel
# (--jobs $(DATASET_JOBS)).
#
# Aggregates run the whole default set:
#
#   make runall-impute     all imputations (plain + --raw)
#   make runall-impute-hier  all imputations through the --hier density descent
#   make runall-factor     all methods (plain + --raw)
#   make runall-loco       leave-one-column-out factor runs (plain + --raw)
#   make runall            everything above, then the loadings comparison
#
# Every job logs to $(LOGS)/<job>.log; one "[ok] <job>" / "[error] <job>" line
# per job goes to $(LOGS)/summary.txt, with the tail of failing logs inlined.
# ------------------------------------------------------------------------------
JOBS ?= 6
TAIL ?= 50
# Hierarchical imputation runs methods SERIALLY (JOBS=1 in the recipe) and
# parallelises independent DATASET chains inside each method instead. DATASET_JOBS
# is the --jobs value handed to impute.R; it is capped by the number of chains.
DATASET_JOBS ?= 8
# Forked chain workers must not each spin up a full BLAS thread pool, and fork
# while OpenBLAS threads are live can deadlock. Pin every math library to 1.
PAR_ENV := OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export JOBS TAIL
LOGS := $(DENSITY_RESULTS)/logs

IMPUTE_METHODS := softimpute onesidedmc missforest knn default zeros
FACTOR_METHODS := softimpute onesidedmc missforest knn default zeros
LOCO_PLAIN     := default zeros softimpute onesidedmc knn   # plain missforest loco skipped

CLEAR_SUMMARY := @mkdir -p $(LOGS) && : > $(LOGS)/summary.txt

# The synth corpus's flat results + database.db live under results/synth/
# regardless of $(RESULTS_ROOT), so its logs live there too (density-nested).
SYNTH_LOGS  := results/synth/$(DENSITY)/logs
CLEAR_SYNTH := @mkdir -p $(SYNTH_LOGS) && : > $(SYNTH_LOGS)/summary.txt

# The method words typed after the target (e.g. "make factor knn missforest").
# "factor-timed" is itself a target, so keep it out of the method-word list.
# Same for "impute-synth".
EXTRA := $(filter-out impute impute-hier runall-impute-hier factor factor-timed impute-synth impute-synth-hier loco,$(MAKECMDGOALS))

.PHONY: deps env env-py env-r env-jl preproc clean \
        impute impute-hier impute-synth impute-synth-hier factor factor-timed loco release-date release-date-report \
        runall runall-impute runall-impute-hier runall-factor runall-loco

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
	uv run python scripts/densify.py --data-root $(DATA_ROOT) --density $(DENSITY)

# Synthetic MNAR stress-test data (scripts/make_mnar.py). Reads
# data/text_only/$(DENSITY) and writes the 4 truth cells x 4 masks grid (s0
# seed) into data/synth/$(DENSITY)/. Each
# data/synth/<density>/<cell>/<mask>/s0 is a drop-in --data-root (or use
# `Rscript src/run/impute.R --synth --density <d>` for all of them at once);
# planted truth (complete matrices + true correlations) lands in
# data/synth/<density>/truth/.
mnar-synth:
	uv run python scripts/make_mnar.py --density $(DENSITY)

# --- ad hoc jobs ---------------------------------------------------------------
impute:
	$(CLEAR_SUMMARY)
	./scripts/runmulti.sh $(LOGS) impute -- $(EXTRA) -- $(ROOTS)
	# ./scripts/runmulti.sh $(LOGS) impute raw -- $(EXTRA) -- $(ROOTS)

# Hierarchical imputation (impute.R --hier): descends the density slices under
# $(DATA_ROOT) DENSEST-FIRST; each step trains on the union grid with the
# previous step's completed columns filled in, and holdouts come only from the
# step's real observations. Unlike the other imputation targets this one runs
# methods SERIALLY (JOBS=1) and lets each method fan its independent dataset
# chains out across $(DATASET_JOBS) workers (--jobs). $(ROOTS) is passed
# verbatim; impute.R parses --density but IGNORES it — hier discovers and sorts
# the densities itself. Outputs land under $(DATA_ROOT)/hierarchical/ and
# $(RESULTS_ROOT)/hierarchical/.
# Default methods: the full IMPUTE_METHODS set (as with `impute`); method
# words after the target override, e.g. "make impute-hier knn".
impute-hier:
	$(CLEAR_SUMMARY)
	JOBS=1 $(PAR_ENV) ./scripts/runmulti.sh $(LOGS) impute hier -- $(if $(EXTRA),$(EXTRA),$(IMPUTE_METHODS)) -- $(ROOTS) --jobs $(DATASET_JOBS)

# Synthetic-corpus imputation (impute.R --synth): one job per method, each
# sweeping all 16 data/synth/$(DENSITY)/<cell>/<mask>/s0 roots. --synth fixes
# its own roots and results layout (flat results + database.db under
# results/synth/$(DENSITY)/, table imputation_synth). Default methods
# missforest + softimpute; method words after the target override, e.g.
#   make impute-synth knn
# Note: runmulti.sh adds --reimpute to impute jobs (same as `make impute`).
impute-synth:
	$(CLEAR_SYNTH)
	./scripts/runmulti.sh $(SYNTH_LOGS) impute synth -- $(if $(EXTRA),$(EXTRA),missforest softimpute) -- --density $(DENSITY)

# Hierarchical imputation over the synth corpus (impute.R --hier --synth): for
# each (cell, mask), descends the data/synth/<density>/.../s0 slices
# densest-first, seeding from the standard --synth run's cached imputed CSV at
# the densest density. Methods run SERIALLY (JOBS=1); each method fans its
# independent (cell, mask, dz, st) chains across $(DATASET_JOBS) workers
# (--jobs). --density is passed but ignored by the descent (it discovers the
# densities itself). Results + DB: flat artifacts under
# results/synth/hierarchical/<cell>/<mask>/<density>/, and ONE unified DB at
# results/synth/hierarchical/database.db (table imputation_synth_hier, keyed by
# cell/mask/density/dataset/method). The final step is handed off to the
# density-0 slice (data/synth/<cell>/<mask>/0/).
impute-synth-hier:
	$(CLEAR_SYNTH)
	JOBS=1 $(PAR_ENV) ./scripts/runmulti.sh $(SYNTH_LOGS) impute synth hier -- $(if $(EXTRA),$(EXTRA),missforest softimpute) -- --density $(DENSITY) --jobs $(DATASET_JOBS)

factor:
	$(CLEAR_SUMMARY)
	./scripts/runmulti.sh $(LOGS) factor -- $(EXTRA) -- $(ROOTS)
	# ./scripts/runmulti.sh $(LOGS) factor raw -- $(EXTRA) -- $(ROOTS)

loco:
	$(CLEAR_SUMMARY)
	./scripts/runmulti.sh $(LOGS) factor loco -- $(EXTRA) -- $(ROOTS)
	# ./scripts/runmulti.sh $(LOGS) factor raw loco -- $(EXTRA) -- $(ROOTS)

# Year-separated factoring: one EFA per release-year cohort of models.
# Fill-smooth methods (default/zeros) are excluded by construction — their
# imputed correlation is global, not per cohort — and the R script
# double-guards against them. Method words after the target override
# the default set, e.g. "make factor-timed knn".
factor-timed:
	$(CLEAR_SUMMARY)
	./scripts/runmulti.sh $(LOGS) factor timed -- $(if $(EXTRA),$(EXTRA),$(FACTOR_METHODS)) -- $(ROOTS)
	# ./scripts/runmulti.sh $(LOGS) factor timed raw -- $(if $(EXTRA),$(EXTRA),$(FACTOR_METHODS)) -- $(ROOTS)

# Release-date analysis (paper appendix "Release-date analysis"). Needs `make
# factor` on the same roots first. Models are binned into release cohorts
# (<=2022, 2023, 2024, >=2025) and each cohort is factored with the row-preserving
# imputers that pass factor.R's R2_GATE, against 50 random subsets of the same
# size. Also: g level per cohort, an imputation-free check on the benchmarks
# every cohort took, and benchmark release year vs the pooled solutions.
# Writes $(DENSITY_RESULTS)/release_date/ (summary.md has every table).
#   make release-date DATA_ROOT=<d> RESULTS_ROOT=<r> DENSITY=<d>
release-date:
	uv run python scripts/release_date.py run $(ROOTS)

# Combine release-date runs (one per target density) into the paper's tables
# and figure. RUNS is a space-separated list of LABEL=<results-root>/release_date.
#   make release-date-report RUNS="10%=results/a/release_date 20%=results/b/release_date"
RUNS ?= $(DENSITY_RESULTS)/release_date
release-date-report:
	uv run python scripts/release_date.py report $(foreach r,$(RUNS),--run $(r)) \
		--out $(DENSITY_RESULTS)/release_date_report

# The method words after the target (e.g. "usvt" in "make factor usvt") arrive as
# goals make wants to build; they're consumed by runmulti via $(EXTRA) above, so
# give them an empty recipe here so make doesn't error "No rule to make target".
.PHONY: $(EXTRA)
$(EXTRA):

# --- aggregates ---------------------------------------------------------------
runall-impute:
	$(CLEAR_SUMMARY)
	./scripts/runmulti.sh $(LOGS) impute -- $(IMPUTE_METHODS) -- $(ROOTS)
	# ./scripts/runmulti.sh $(LOGS) impute raw -- $(IMPUTE_METHODS) -- $(ROOTS)

# All imputations through the hierarchical density descent (impute.R --hier):
# seeds from the densest density's standard-run caches, then fills downward.
# Methods serial; each method's independent chains run $(DATASET_JOBS)-wide.
runall-impute-hier:
	$(CLEAR_SUMMARY)
	JOBS=1 $(PAR_ENV) ./scripts/runmulti.sh $(LOGS) impute hier -- $(IMPUTE_METHODS) -- $(ROOTS) --jobs $(DATASET_JOBS)

runall-factor:
	$(CLEAR_SUMMARY)
	./scripts/runmulti.sh $(LOGS) factor -- $(FACTOR_METHODS) -- $(ROOTS)
	# ./scripts/runmulti.sh $(LOGS) factor raw -- $(FACTOR_METHODS) -- $(ROOTS)

runall-loco:
	$(CLEAR_SUMMARY)
	./scripts/runmulti.sh $(LOGS) factor loco -- $(LOCO_PLAIN) -- $(ROOTS)
	# ./scripts/runmulti.sh $(LOGS) factor raw loco -- $(FACTOR_METHODS) -- $(ROOTS)

runall:
	$(CLEAR_SUMMARY)
	$(MAKE) --no-print-directory -j$(JOBS) runall-impute runall-factor runall-loco
	./scripts/runone.sh compare-loadings $(LOGS) uv run python scripts/compare_loadings.py --results $(RESULTS_ROOT) --density $(DENSITY)

clean:
	cd results && rm -rf *
