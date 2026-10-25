FROM ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake ccache git ca-certificates python3 && rm -rf /var/lib/apt/lists/*
ENV CCACHE_DIR=/ccache
WORKDIR /work
