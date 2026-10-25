FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends git curl ca-certificates python3 \
    && rm -rf /var/lib/apt/lists/*

# Only the base commit is fetched; the reference commit and everything after it never enter the image.
RUN git init -q /app && cd /app \
    && git remote add origin https://github.com/a8nova/adreno-llms \
    && git fetch -q --depth 1 origin f6e489b3326b94264dde6e78ea5c7a68c748a258 && git checkout -q FETCH_HEAD \
    && git remote remove origin \
    && printf 'task/\n' >> .git/info/exclude

COPY task /app/task
RUN chmod +x /app/task/phone_run.sh
WORKDIR /app
