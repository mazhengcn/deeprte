# First, build the application in the `/deeprte` directory
FROM ghcr.io/astral-sh/uv:debian-slim AS builder
# Install system packages
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
# Configure the Python directory so it is consistent
ENV UV_PYTHON_INSTALL_DIR=/python
# Only use the managed Python version
ENV UV_PYTHON_PREFERENCE=only-managed

WORKDIR /deeprte
# Install Python before the project for caching
RUN --mount=type=bind,source=.python-version,target=.python-version \
    uv python install
# Install the project dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --all-extras --no-dev
# Install the project
COPY . /deeprte
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --all-extras --no-dev

# Then, use a final image without uv
FROM debian:bookworm-slim

# Copy the Python version
COPY --from=builder --chown=python:python /python /python

# Copy the application from the builder
COPY --from=builder --chown=deeprte:deeprte /deeprte /deeprte

# Place executables in the environment at the front of the path
ENV PATH="/deeprte/.venv/bin:$PATH"

WORKDIR /deeprte
