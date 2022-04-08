# [Choice] CUDA and Python version: 11.4-3.9
ARG VARIANT=3.10-11.6
FROM zhengma/python-3-cuda:${VARIANT}

# [Optional] If your pip requirements rarely change, uncomment this section to add them to the image.
COPY requirements.txt /tmp/pip-tmp/
RUN python -m pip --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements.txt
