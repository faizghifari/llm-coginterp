JULIA_PROJECT := src/impute/OneSidedMC

.PHONY: install install-py install-r install-julia

# Install all three environments: Python (uv), R (install.R), Julia (OSMC project).
install: install-py install-r install-julia
	@echo "All environments installed."

install-py:
	uv sync

install-r:
	Rscript install.R

install-julia:
	julia --project=$(JULIA_PROJECT) -e 'using Pkg; Pkg.instantiate()'
