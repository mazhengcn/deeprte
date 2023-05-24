FROM python:3.11-slim-bullseye AS base

FROM base AS pip-installs
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/deeprteenv
ENV PATH="/opt/deeprteenv/bin:$PATH"

COPY requirements.txt /tmp/pip-tmp/
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /tmp/pip-tmp/requirements.txt \
    && rm -rf /tmp/pip-tmp

FROM base
COPY --from=pip-installs /opt/deeprteenv /opt/deeprteenv
ENV PATH="/opt/deeprteenv/bin:$PATH"
