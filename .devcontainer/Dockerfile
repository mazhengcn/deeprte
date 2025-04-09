ARG PYTHON_VERSION=3.12

FROM mcr.microsoft.com/devcontainers/python:$PYTHON_VERSION

RUN apt-get update && apt-get install -y --no-install-recommends screen \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest --chown=vscode: /uv /uvx /bin/
