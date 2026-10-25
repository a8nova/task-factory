# Task Factory cloud gateway: the trusted grader, on Cloud Run.
# Builds adreno-llms ports with the Android NDK and times them on Google's Device Run phones.
FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake make ccache curl unzip ca-certificates gnupg file \
    && curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
        > /etc/apt/sources.list.d/google-cloud-sdk.list \
    && apt-get update && apt-get install -y --no-install-recommends google-cloud-cli \
    && rm -rf /var/lib/apt/lists/*

# Android NDK r26d, the version the adreno-llms build scripts were written against.
RUN curl -fsSL -o /tmp/ndk.zip https://dl.google.com/android/repository/android-ndk-r26d-linux.zip \
    && unzip -q /tmp/ndk.zip -d /opt && mv /opt/android-ndk-r26d /opt/ndk && rm /tmp/ndk.zip
ENV ANDROID_NDK=/opt/ndk

# Builds run as this unprivileged user; the server (root) keeps every secret in memory only.
RUN useradd --create-home --uid 10001 builder && test -x /usr/sbin/runuser

# The source repo (root-only) and a local CLBlast mirror for FetchContent.
RUN git clone -q -c core.autocrlf=false https://github.com/a8nova/adreno-llms /opt/adreno-llms \
    && git -C /opt/adreno-llms config core.autocrlf false \
    && chmod 700 /opt/adreno-llms \
    && git clone -q --mirror https://github.com/CNugteren/CLBlast.git /opt/mirrors/CLBlast.git

USER builder
ENV HOME=/home/builder
# OpenCL headers + link stub the way the repo's own device-free CI makes them, at both paths the
# task commits' build scripts look in.
RUN mkdir -p /home/builder/.cache /home/builder/.nnopt \
    && git clone -q https://github.com/a8nova/adreno-llms /tmp/al \
    && ADRENO_LLMS_CACHE=/home/builder/.cache/adreno-llms bash /tmp/al/scripts/ci/setup_opencl_stub.sh > /tmp/stub.log 2>&1 \
    && ln -s /home/builder/.cache/adreno-llms /home/builder/.nnopt/deps \
    && rm -rf /tmp/al \
    && git config --global url."file:///opt/mirrors/CLBlast.git".insteadOf https://github.com/CNugteren/CLBlast.git \
    && git config --global --add safe.directory '*'
USER root

COPY app /app
WORKDIR /app

# Pre-warm the compiler cache with the base and reference commits of this task family.
RUN python3 -m builder prewarm smollm2-135m-instruct \
        f6e489b3326b94264dde6e78ea5c7a68c748a258 e2076eeb9b20ec6b79449fb6db7f86e80d00ba47 \
        4f0f2af972fb08361524964f9b826a64fa5e4817 887bd7e927e22374c56126fbc03bcb552b418481 \
    && for port in openelm-270m qwen2-5-0-5b lfm2-5-350m; do \
         python3 -m builder prewarm $port 51fe88ab63 4f0f2af972fb08361524964f9b826a64fa5e4817 || exit 1; done \
    && python3 -m builder prewarm lfm2-5-350m \
        f8beda1ab7f5554dd1a8a4fcfd156bdd2185b596 887bd7e927e22374c56126fbc03bcb552b418481

ENV PORT=8080
CMD ["python3", "-u", "server.py"]
