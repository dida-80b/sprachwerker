ARG BASE_IMAGE=rocm/dev-ubuntu-22.04:6.3.4-complete
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    ffmpeg \
    libsndfile1 \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /app/requirements.txt

COPY asr_api.py /app/asr_api.py
COPY scripts /app/scripts
COPY templates /app/templates
COPY static /app/static

ENV PORT=8095

CMD ["python3", "/app/asr_api.py"]
