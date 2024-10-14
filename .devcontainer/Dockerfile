FROM debian:bookworm-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates git wget \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=True \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/deeprte/.venv \
    PATH="/deeprte/.venv/bin:$PATH"

WORKDIR /deeprte
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=.python-version,target=.python-version \
    uv sync --no-install-project --no-dev

ADD https://starship.rs/install.sh /starship-installer.sh
RUN sh /starship-installer.sh -y && rm /starship-installer.sh \
    && echo "eval '$(starship init bash)'" >> /root/.bashrc
