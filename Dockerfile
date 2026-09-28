FROM python:3.11-slim

# INSTALL_ML=1 adds torch + ultralytics + cellpose for the learned engines.
ARG INSTALL_ML=0
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements-ml.txt ./
RUN pip install -r requirements.txt \
    && if [ "$INSTALL_ML" = "1" ]; then pip install -r requirements-ml.txt; fi

COPY colony_detector ./colony_detector
COPY tools ./tools
COPY models ./models

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"
CMD ["uvicorn", "colony_detector.service:app", "--host", "0.0.0.0", "--port", "8000"]
