ARG WEFTBENCH_DEPS_IMAGE=python:3.11-slim
FROM ${WEFTBENCH_DEPS_IMAGE} AS dependencies
USER root
WORKDIR /build
COPY requirements.txt .
RUN python -m pip install --disable-pip-version-check --no-cache-dir --only-binary=:all: -r requirements.txt && python -m pip check

FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HOST=0.0.0.0 PORT=8000 WEFTBENCH_DATA=/var/lib/weftbench
WORKDIR /opt/weftbench
COPY --from=dependencies /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY weftbench weftbench
COPY static static
RUN useradd -U -u 11504 weftbench && install -d -o weftbench -g weftbench /var/lib/weftbench
USER weftbench
EXPOSE 8000
CMD ["python", "-m", "weftbench"]
