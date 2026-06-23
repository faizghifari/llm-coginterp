JULIA_PROJECT := src/impute/OneSidedMC

.PHONY: install env-py env-r env-julia

SUDO := $(shell if [ "$$(id -u)" -eq 0 ]; then echo ""; else echo "sudo"; fi)

deps:
	$(SUDO) apt update -y
	$(SUDO) apt install r-base -y
	curl -fsSL https://install.julialang.org | sh -s -- -y
	curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all three environments: Python (uv), R (install.R), Julia (OSMC project).
env: env-py env-r env-julia
	@echo "All environments installed."


export PATH := $(HOME)/.local/bin:$(PATH)

env-py:
	uv sync

env-r:
	Rscript install.R

env-julia:
	julia --project=$(JULIA_PROJECT) -e 'using Pkg; Pkg.instantiate()'

preproc:
	python scripts/collapse_results.py
	python scripts/densify.py

runall:
	# skip iterativepca (not pushed to git) and mice (slow as hell)
	# skip sensitivity, probably not needed
	Rscript src/run/main.R --method softimpute --reimpute
	Rscript src/run/main.R --method softimpute --reimpute  --raw

	Rscript src/run/main.R --method onesidedmc --reimpute
	Rscript src/run/main.R --method onesidedmc --reimpute --raw

	Rscript src/run/main.R --method missforest --reimpute
	Rscript src/run/main.R --method missforest --reimpute --raw

	Rscript src/run/main.R --method knn --reimpute
	Rscript src/run/main.R --method knn --reimpute --raw

	python scripts/compare_loadings.py


clean:
	cd results && rm -rf *
