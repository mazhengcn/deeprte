ARG UV_VERSION=latest

FROM ghcr.io/astral-sh/uv:$UV_VERSION AS uv

FROM debian:trixie-slim

# Install system packages
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends screen sudo git gh ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=True \
    PYTHONUNBUFFERED=True \
    UV_LINK_MODE=copy

# Create nonroot user
ARG USERNAME=deeprte
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID --shell /bin/bash --create-home $USERNAME \
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

COPY --from=uv --chown=deeprte:deeprte /uv /uvx /bin/
