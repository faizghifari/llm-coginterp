FROM ubuntu:resolute-20260610

RUN apt-get update && apt-get install -y make curl r-base
RUN curl -fsSL https://install.julialang.org | sh -s -- -y
RUN	curl -LsSf https://astral.sh/uv/install.sh | sh

COPY . .

RUN make env-py
RUN make env-jl
RUN make env-r
