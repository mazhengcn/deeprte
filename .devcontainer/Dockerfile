FROM debian:bookworm-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates git wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /deeprte

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_PROJECT_ENVIRONMENT=/deeprte/.venv

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=.python-version,target=.python-version \
    uv sync --no-install-project --no-dev

ENV PATH="/deeprte/.venv/bin:$PATH"

ADD https://starship.rs/install.sh /starship-installer.sh
RUN sh /starship-installer.sh -y && rm /starship-installer.sh \
    && echo "eval '$(starship init bash)'" >> /root/.bashrc
