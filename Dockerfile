FROM ubuntu:resolute-20260610

RUN apt-get update && apt-get install -y make curl r-base sudo
RUN curl -fsSL https://install.julialang.org | sh -s -- -y
RUN	curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /app
COPY . .

RUN make env-py
RUN make env-jl

RUN apt-get update && apt-get install -y g++ gfortran libblas-dev liblapack-dev zlib1g-dev cmake libxml2-dev rustup xz-utils libgmp-dev libmpfr-dev wget \
&& rustup default stable

RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common dirmngr \
 && wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | tee -a /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc \
 && add-apt-repository "deb https://cloud.r-project.org/bin/linux/ubuntu resolute-cran40/" \
 && apt-get update && apt-get install -y r-base r-base-dev

RUN apt-get update && apt-get install -y libfreetype6-dev libuv1-dev

RUN make env-r
