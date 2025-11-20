FROM debian:trixie-slim

# Install system packages
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends screen sudo git gh ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV UV_LINK_MODE=copy
# Only use the managed Python version
ENV UV_PYTHON_PREFERENCE=only-managed
# Configure the Python directory so it is consistent
ENV UV_PYTHON_INSTALL_DIR=/python

WORKDIR /python
RUN --mount=type=bind,source=.python-version,target=.python-version \
    uv python install

# Create nonroot user
ARG USERNAME=deeprte
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID --shell /bin/bash --create-home $USERNAME \
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME
USER $USERNAME
