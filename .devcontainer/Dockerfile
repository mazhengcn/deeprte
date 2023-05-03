# JAX and Python
ARG JAX_VERSION=0.4.8
ARG PYTHON_VERSION=3.11
ARG VENV_PATH=/opt/deeprteenv
# CUDA verions
ARG CUDA_MAJOR_VERSION=12

FROM python:${PYTHON_VERSION}-slim-bullseye AS pip-installs
ARG JAX_VERSION
ARG CUDA_MAJOR_VERSION
ARG VENV_PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv ${VENV_PATH}
ENV PATH=${VENV_PATH}/bin:$PATH
RUN pip install --no-cache-dir --upgrade pip wheel \
    && pip install --no-cache-dir -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
    absl-py \
    jax[cuda${CUDA_MAJOR_VERSION}_pip]==${JAX_VERSION} \
    jaxline@git+https://github.com/mazhengcn/jaxline \
    dm-haiku \
    ml-collections \
    optax \
    tensorflow-cpu \
    tensorflow-datasets

FROM python:${PYTHON_VERSION}-slim-bullseye
ARG VENV_PATH
# Python packages
COPY --from=pip-installs ${VENV_PATH} ${VENV_PATH}
ENV PATH=${VENV_PATH}/bin:$PATH
