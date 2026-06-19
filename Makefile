JULIA_PROJECT := src/impute/OneSidedMC

.PHONY: install install-py install-r install-julia

deps:
	curl -fsSL https://install.julialang.org | sh
	curl -LsSf https://astral.sh/uv/install.sh | sh

	sudo apt update -y
	sudo apt install r-base -y

# Install all three environments: Python (uv), R (install.R), Julia (OSMC project).
install: install-py install-r install-julia
	@echo "All environments installed."

install-py:
	uv sync

install-r:
	Rscript install.R

install-julia:
	julia --project=$(JULIA_PROJECT) -e 'using Pkg; Pkg.instantiate()'

preproc:
	python scripts/collapse_results.py
	python scripts/densify.py

runall:
	# skip iterativepca (not pushed to git) and mice (slow as hell)
	Rscript src/run/main.R --method softimpute --reimpute --sensitivity
	Rscript src/run/main.R --method softimpute --reimpute  --sensitivity --raw

	Rscript src/run/main.R --method onesidedmc --reimpute --sensitivity
	Rscript src/run/main.R --method onesidedmc --reimpute --sensitivity --raw

	Rscript src/run/main.R --method missforest --reimpute --sensitivity
	Rscript src/run/main.R --method missforest --reimpute --sensitivity --raw

	Rscript src/run/main.R --method knn --reimpute --sensitivity
	Rscript src/run/main.R --method knn --reimpute --sensitivity --raw

	python scripts/compare_loadings.py


clean:
	cd results && rm -rf *
