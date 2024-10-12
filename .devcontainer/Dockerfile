FROM mcr.microsoft.com/vscode/devcontainers/base:ubuntu-24.04
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app/deeprte

ENV PYTHONUNBUFFERED=True
ENV UV_LINK_MODE=copy
ENV UV_PROJECT_ENVIRONMENT=/app/deeprte/.venv

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=.python-version,target=.python-version \
    uv sync --no-install-project --no-dev

ENV PATH="/app/deeprte/.venv/bin:$PATH"
