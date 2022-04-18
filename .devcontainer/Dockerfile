# syntax=docker/dockerfile:1

ARG VARIANT="py310-cu116"
FROM zhengma/jax:${VARIANT}

# Install some develop tools.
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get install -y --no-install-recommends git wget screen \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

COPY --from=zhengma/base:focal /root/.bashrc /root/.profile /root/

# [Optional] If your pip requirements rarely change, uncomment this section to add them to the image.
# COPY requirements.txt /tmp/pip-tmp/
# RUN python -m pip --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements.txt
