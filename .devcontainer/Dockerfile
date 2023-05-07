FROM python:3.11-slim-bullseye AS base

FROM base AS pip-installs
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/deeprteenv
ENV PATH="/opt/deeprteenv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip wheel \
    && pip install --no-cache-dir \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
    absl-py \
    jax[cuda12_pip] \
    jaxline@git+https://github.com/mazhengcn/jaxline \
    dm-haiku \
    ml-collections \
    optax \
    tensorflow-cpu \
    tensorflow-datasets

FROM base
COPY --from=pip-installs /opt/deeprteenv /opt/deeprteenv

ENV PATH="/opt/deeprteenv/bin:$PATH"
