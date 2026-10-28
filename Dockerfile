FROM python:3.12-slim

# Prevent python from writing .pyc and buffer logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
  && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
# Increase pip network timeout/retries (helps with large wheels on slow connections)
ENV PIP_DEFAULT_TIMEOUT=900

# Silero VAD for /ws/voice endpointing, pinned to a release and checksum so a
# changed upstream file fails the build instead of silently shipping.
ARG SILERO_VAD_TAG=v6.2.3
ARG SILERO_VAD_SHA256=1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3
RUN mkdir -p /app/models \
  && curl -fsSL -o /app/models/silero_vad.onnx \
     "https://raw.githubusercontent.com/snakers4/silero-vad/${SILERO_VAD_TAG}/src/silero_vad/data/silero_vad.onnx" \
  && echo "${SILERO_VAD_SHA256}  /app/models/silero_vad.onnx" | sha256sum -c -

COPY . /app/

ENV THURSDAY_BRAIN_DIR=/app/brain

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
