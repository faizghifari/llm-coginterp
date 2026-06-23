FROM ubuntu:resolute-20260610

COPY . .

RUN apt-get update && apt-get install -y make curl
RUN make deps
RUN make env
