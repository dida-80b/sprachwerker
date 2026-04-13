FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir piper-phonemize || true
RUN pip install --no-cache-dir piper-tts --no-deps
RUN pip install --no-cache-dir \
        onnxruntime-rocm \
        --index-url https://repo.radeon.com/rocm/manylinux/rocm-rel-6.2/ \
    || pip install --no-cache-dir onnxruntime
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .

RUN mkdir -p /voices
ADD https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/thorsten/high/de_DE-thorsten-high.onnx /voices/de_DE-thorsten-high.onnx
ADD https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/thorsten/high/de_DE-thorsten-high.onnx.json /voices/de_DE-thorsten-high.onnx.json
RUN echo "9df1c43c61149ef9b39e618e2b861fbe41e1fcea9390b2dac62e8761573ea4f1  /voices/de_DE-thorsten-high.onnx" | sha256sum -c -
RUN echo "6de734444e4c3f9e33b7ebe2746dbc19b71e85f613e79c65acf623200b99a76a  /voices/de_DE-thorsten-high.onnx.json" | sha256sum -c -

ENV HSA_OVERRIDE_GFX_VERSION=11.0.0
ENV VOICES_DIR=/voices
ENV DEFAULT_VOICE=de_DE-thorsten-high
ENV PORT=5150

EXPOSE 5150

CMD ["python3", "server.py"]
