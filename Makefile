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

.PHONY: install env-py env-r env-jl

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

runall:
	# skip iterativepca (not pushed to git) and mice (slow as hell)
	# skip sensitivity, probably not needed
	Rscript src/run/impute.R --method softimpute --reimpute $(ROOTS)
	Rscript src/run/impute.R --method softimpute --reimpute  --raw $(ROOTS)

	Rscript src/run/impute.R --method onesidedmc --reimpute $(ROOTS)
	Rscript src/run/impute.R --method onesidedmc --reimpute --raw $(ROOTS)

	Rscript src/run/impute.R --method missforest --reimpute $(ROOTS)
	Rscript src/run/impute.R --method missforest --reimpute --raw $(ROOTS)

	Rscript src/run/impute.R --method knn --reimpute $(ROOTS)
	Rscript src/run/impute.R --method knn --reimpute --raw $(ROOTS)

	Rscript src/run/factor.R --method softimpute $(ROOTS)
	Rscript src/run/factor.R --method softimpute --raw $(ROOTS)

	Rscript src/run/factor.R --method onesidedmc $(ROOTS)
	Rscript src/run/factor.R --method onesidedmc --raw $(ROOTS)

	Rscript src/run/factor.R --method missforest $(ROOTS)
	Rscript src/run/factor.R --method missforest --raw $(ROOTS)

	Rscript src/run/factor.R --method knn $(ROOTS)
	Rscript src/run/factor.R --method knn --raw $(ROOTS)

	uv run python scripts/compare_loadings.py --results $(RESULTS_ROOT)

runall-impute:
	Rscript src/run/impute.R --method softimpute --reimpute $(ROOTS)
	Rscript src/run/impute.R --method softimpute --reimpute --raw $(ROOTS)

	Rscript src/run/impute.R --method onesidedmc --reimpute $(ROOTS)
	Rscript src/run/impute.R --method onesidedmc --reimpute --raw $(ROOTS)

	Rscript src/run/impute.R --method missforest --reimpute $(ROOTS)
	Rscript src/run/impute.R --method missforest --reimpute --raw $(ROOTS)

	Rscript src/run/impute.R --method knn --reimpute $(ROOTS)
	Rscript src/run/impute.R --method knn --reimpute --raw $(ROOTS)

runall-factor:
	Rscript src/run/factor.R --method raw $(ROOTS)

	Rscript src/run/factor.R --method softimpute $(ROOTS)
	Rscript src/run/factor.R --method softimpute --raw $(ROOTS)

	Rscript src/run/factor.R --method onesidedmc $(ROOTS)
	Rscript src/run/factor.R --method onesidedmc --raw $(ROOTS)

	Rscript src/run/factor.R --method missforest $(ROOTS)
	Rscript src/run/factor.R --method missforest --raw $(ROOTS)

	Rscript src/run/factor.R --method knn $(ROOTS)
	Rscript src/run/factor.R --method knn --raw $(ROOTS)

runall-loco:
	Rscript src/run/factor.R --method raw --loco $(ROOTS)

	Rscript src/run/factor.R --method softimpute --loco $(ROOTS)
	Rscript src/run/factor.R --method softimpute --raw --loco $(ROOTS)

	Rscript src/run/factor.R --method onesidedmc --loco $(ROOTS)
	Rscript src/run/factor.R --method onesidedmc --raw --loco $(ROOTS)

	# Rscript src/run/factor.R --method missforest --loco $(ROOTS)
	Rscript src/run/factor.R --method missforest --raw --loco $(ROOTS)

	Rscript src/run/factor.R --method knn --loco $(ROOTS)
	Rscript src/run/factor.R --method knn --raw --loco $(ROOTS)

clean:
	cd results && rm -rf *
