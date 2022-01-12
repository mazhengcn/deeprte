# [Choice] CUDA and Python version: 11.4-3.9
ARG VARIANT=3.10-11.5
FROM zhengma/python-3-cuda:${VARIANT}

# Set time zone
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone

# [Optional] If your pip requirements rarely change, uncomment this section to add them to the image.
COPY requirements-dev.txt jaxline.patch /tmp/pip-tmp/
# Temporary hack to build dm-tree from source in Python 3.10, so we need to install cmake first.
# RUN pip3 --disable-pip-version-check --no-cache-dir install cmake
RUN pip3 --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements-dev.txt

# Apply JAXline and tensorboard patch
WORKDIR /usr/local/python/lib/python3.10/site-packages
RUN patch -p 0 < /tmp/pip-tmp/jaxline.patch \
    && rm -rf /tmp/pip-tmp
