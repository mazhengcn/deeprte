# PYTHON_VERSION
ARG PYTHON_VERSION=3.12

FROM mcr.microsoft.com/devcontainers/base:debian
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends screen \
    && rm -rf /var/lib/apt/lists/*
ADD https://starship.rs/install.sh /starship-installer.sh
RUN sh /starship-installer.sh -y && rm /starship-installer.sh \
    && echo 'eval "$(starship init bash)"' >> /home/vscode/.bashrc

COPY --from=ghcr.io/astral-sh/uv:latest --chown=vscode: /uv /uvx /bin/
ENV UV_LINK_MODE=copy
ENV UV_PYTHON_INSTALL_DIR=/python
ENV UV_PYTHON_PREFERENCE=only-managed

ARG PYTHON_VERSION
RUN uv python install ${PYTHON_VERSION}
