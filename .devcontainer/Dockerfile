FROM rust:latest AS rust-builder

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt update && apt-get --no-install-recommends install -y \
    cmake \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/usr/local/cargo/git/db \
    --mount=type=cache,target=/usr/local/cargo/registry/ \
    cargo install --git https://github.com/astral-sh/uv uv \
    && cargo install starship

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    openssh-client \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY --from=rust-builder /usr/local/cargo/bin/uv \
    /usr/local/cargo/bin/uvx \
    /usr/local/cargo/bin/starship \
    /usr/local/bin/

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_PROJECT_ENVIRONMENT=/deeprte/.venv

WORKDIR /deeprte
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=.python-version,target=.python-version \
    uv sync --no-install-project --no-dev
ENV PATH="/deeprte/.venv/bin:$PATH"

RUN echo 'eval "$(uv generate-shell-completion bash)"' >> ~/.bashrc \
    && echo 'eval "$(uvx --generate-shell-completion bash)"' >> ~/.bashrc \
    && echo 'eval "$(starship init bash)"' >> ~/.bashrc
