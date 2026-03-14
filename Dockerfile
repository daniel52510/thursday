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
COPY . /app/

ENV THURSDAY_BRAIN_DIR=/app/brain

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
