# JAX and Python
ARG JAX_VERSION=0.4.7
ARG PYTHON_VERSION=3.11
ARG VENV_PATH=/opt/deeprteenv
# CUDA verions
ARG CUDA_MAJOR_VERSION=12
ARG CUDA_MINOR_VERSION=0
ARG CUDA_PATCH_VERSION=1
ARG CUDA_VERSION=${CUDA_MAJOR_VERSION}.${CUDA_MINOR_VERSION}
ARG CUDNN_MAJOR_VERSION=8
# CUDA library versions
ARG CUFFT_MAJOR_VERSION=11
ARG CURAND_MAJOR_VERSION=10
ARG CUSOLVER_MAJOR_VERSION=11

FROM nvidia/cuda:${CUDA_VERSION}.${CUDA_PATCH_VERSION}-cudnn${CUDNN_MAJOR_VERSION}-devel-ubuntu22.04 AS cuda-libs

FROM python:${PYTHON_VERSION}-slim AS pip-installs
ARG JAX_VERSION
ARG VENV_PATH
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*
RUN python -m venv ${VENV_PATH}
ENV PATH=${VENV_PATH}/bin:$PATH
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
    absl-py \
    jax[cuda11_cudnn88]==${JAX_VERSION} \
    jaxline@git+https://github.com/mazhengcn/jaxline \
    dm-haiku \
    ml-collections \
    numpy \
    optax \
    tensorflow-cpu \
    tensorflow-datasets

FROM python:${PYTHON_VERSION}-slim
ARG PYTHON_VERSION
ARG VENV_PATH
ARG CUDA_MAJOR_VERSION
ARG CUDA_VERSION
ARG CUDNN_MAJOR_VERSION
ARG CUFFT_MAJOR_VERSION
ARG CURAND_MAJOR_VERSION
ARG CUSOLVER_MAJOR_VERSION

# Python packages
COPY --from=pip-installs ${VENV_PATH} ${VENV_PATH}
# CUDA libraries
COPY --from=cuda-libs /usr/local/cuda/compat /usr/local/cuda/compat
COPY --from=cuda-libs /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1
COPY --from=cuda-libs \
    /usr/local/cuda/lib64/libcublas.so.${CUDA_MAJOR_VERSION} \
    /usr/local/cuda/lib64/libcublasLt.so.${CUDA_MAJOR_VERSION} \
    /usr/local/cuda/lib64/libcudart.so.${CUDA_MAJOR_VERSION} \
    /usr/local/cuda/lib64/libcufft.so.${CUFFT_MAJOR_VERSION} \
    /usr/local/cuda/lib64/libcufftw.so.${CUFFT_MAJOR_VERSION} \
    /usr/local/cuda/lib64/libcuinj64.so.${CUDA_VERSION} \
    /usr/local/cuda/lib64/libcupti.so.${CUDA_MAJOR_VERSION} \
    /usr/local/cuda/lib64/libcurand.so.${CURAND_MAJOR_VERSION} \
    /usr/local/cuda/lib64/libcusolver.so.${CUSOLVER_MAJOR_VERSION} \
    /usr/local/cuda/lib64/libcusolverMg.so.${CUSOLVER_MAJOR_VERSION} \
    /usr/local/cuda/lib64/libcusparse.so.${CUDA_MAJOR_VERSION} \
    /usr/local/cuda/lib64/libnvrtc-builtins.so.${CUDA_VERSION} \
    /usr/local/cuda/lib64/libnvrtc.so.${CUDA_MAJOR_VERSION} \
    /usr/local/cuda/lib64/
COPY --from=cuda-libs \
    /usr/lib/x86_64-linux-gnu/libcudnn.so.${CUDNN_MAJOR_VERSION} \
    /usr/lib/x86_64-linux-gnu/libcudnn_adv_infer.so.${CUDNN_MAJOR_VERSION} \
    /usr/lib/x86_64-linux-gnu/libcudnn_adv_train.so.${CUDNN_MAJOR_VERSION} \
    /usr/lib/x86_64-linux-gnu/libcudnn_cnn_infer.so.${CUDNN_MAJOR_VERSION} \
    /usr/lib/x86_64-linux-gnu/libcudnn_cnn_train.so.${CUDNN_MAJOR_VERSION} \
    /usr/lib/x86_64-linux-gnu/libcudnn_ops_infer.so.${CUDNN_MAJOR_VERSION} \
    /usr/lib/x86_64-linux-gnu/libcudnn_ops_train.so.${CUDNN_MAJOR_VERSION} \
    /usr/lib/x86_64-linux-gnu/
COPY --from=cuda-libs /usr/local/cuda/bin/ptxas /usr/local/cuda/bin/ptxas

ENV LC_ALL=C.UTF-8
ENV PATH=${VENV_PATH}/bin:/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility