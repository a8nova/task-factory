FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends git curl ca-certificates python3 \
    && rm -rf /var/lib/apt/lists/*

# Only the base commit is fetched; the reference commit and everything after it never enter the image.
RUN git init -q /workspace/repo && cd /workspace/repo \
    && git remote add origin https://github.com/a8nova/adreno-llms \
    && git fetch -q --depth 1 origin {{BASE}} && git checkout -q FETCH_HEAD \
    && git remote remove origin \
    && git config --global user.email agent@task-factory.local && git config --global user.name agent \
    && git config --global --add safe.directory /workspace/repo

COPY workspace/ /workspace/
RUN mkdir -p /workspace/submission && chmod +x /workspace/validation/val.sh /workspace/timer.sh
WORKDIR /workspace
