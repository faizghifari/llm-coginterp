JULIA_PROJECT := src/impute/OneSidedMC

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
	uv run python scripts/collapse_results.py
	uv run python scripts/densify.py

runall:
	# skip iterativepca (not pushed to git) and mice (slow as hell)
	# skip sensitivity, probably not needed
	Rscript src/run/impute.R --method softimpute --reimpute
	Rscript src/run/impute.R --method softimpute --reimpute  --raw

	Rscript src/run/impute.R --method onesidedmc --reimpute
	Rscript src/run/impute.R --method onesidedmc --reimpute --raw

	Rscript src/run/impute.R --method missforest --reimpute
	Rscript src/run/impute.R --method missforest --reimpute --raw

	Rscript src/run/impute.R --method knn --reimpute
	Rscript src/run/impute.R --method knn --reimpute --raw

	Rscript src/run/factor.R --method softimpute
	Rscript src/run/factor.R --method softimpute --raw

	Rscript src/run/factor.R --method onesidedmc
	Rscript src/run/factor.R --method onesidedmc --raw

	Rscript src/run/factor.R --method missforest
	Rscript src/run/factor.R --method missforest --raw

	Rscript src/run/factor.R --method knn
	Rscript src/run/factor.R --method knn --raw

	uv run python scripts/compare_loadings.py

runall-impute:
	Rscript src/run/impute.R --method softimpute --reimpute
	Rscript src/run/impute.R --method softimpute --reimpute --raw

	Rscript src/run/impute.R --method onesidedmc --reimpute
	Rscript src/run/impute.R --method onesidedmc --reimpute --raw

	Rscript src/run/impute.R --method missforest --reimpute
	Rscript src/run/impute.R --method missforest --reimpute --raw

	Rscript src/run/impute.R --method knn --reimpute
	Rscript src/run/impute.R --method knn --reimpute --raw

runall-factor:
	Rscript src/run/factor.R --method softimpute
	Rscript src/run/factor.R --method softimpute --raw

	Rscript src/run/factor.R --method onesidedmc
	Rscript src/run/factor.R --method onesidedmc --raw

	Rscript src/run/factor.R --method missforest
	Rscript src/run/factor.R --method missforest --raw

	Rscript src/run/factor.R --method knn
	Rscript src/run/factor.R --method knn --raw

clean:
	cd results && rm -rf *
