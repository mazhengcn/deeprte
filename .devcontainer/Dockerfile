# [Choice] CUDA and Python version: 11.4-3.9
ARG VARIANT=11.4-3.9
FROM zhengma/cuda-python:${VARIANT}

# [Optional] If your pip requirements rarely change, uncomment this section to add them to the image.
COPY requirements-dev.txt /tmp/pip-tmp/
RUN pip3 --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements-dev.txt \
    && rm -rf /tmp/pip-tmp
