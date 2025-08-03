FROM python:3.11-slim-bullseye AS base
RUN sed -i s@/deb.debian.org/@/mirrors.aliyun.com/@g /etc/apt/sources.list

FROM base AS pip-installs
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip --index-url https://mirrors.aliyun.com/pypi/simple \
    && pip install --index-url https://mirrors.aliyun.com/pypi/simple -r /tmp/requirements.txt \
    && rm -rf /tmp/requirements.txt

FROM base
COPY --from=pip-installs /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"